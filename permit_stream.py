"""Test client for resuming interrupted tool calls via POST /stream."""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx

from agents.general_agent import GENERAL_AGENT_ID
from schemas.invoke_request import DecisionType, Permission
from stream import (
    _print_chunk_summary,
    _print_final_summary,
    accumulate_root_reply,
    stream_agent,
    stream_status_from_chunk,
)

DEFAULT_THREAD_ID = "915491e2-2060-41da-8e3e-6e1e629993ec"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MAX_ROUNDS = 10
DEFAULT_PERMISSIONS: list[Permission] = [
    {
        "name": "web_search_tool",
        "decision": DecisionType.APPROVE,
    }
]


def _summarize_action_requests(action_requests: list[dict]) -> str:
    """Return a short human-readable summary of pending tool actions."""
    if not action_requests:
        return "none"
    lines: list[str] = []
    for index, action_request in enumerate(action_requests, start=1):
        name = action_request.get("name", "unknown")
        args = action_request.get("args", {})
        query = args.get("query")
        if query is not None:
            lines.append(f"{index}. {name}: {query!r}")
        else:
            lines.append(f"{index}. {name}: {args!r}")
    return "\n  ".join(lines)


def permit_stream_round(
    *,
    thread_id: str,
    permissions: list[Permission],
    agent_id: int = GENERAL_AGENT_ID,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 300.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Resume an interrupted thread via /stream and accumulate streamed text."""
    chunk_count = 0
    accumulated_text = ""
    awaiting_tool_permission = False
    action_requests: list[dict[str, Any]] = []
    run_status: str | None = None

    for chunk in stream_agent(
        base_url=base_url,
        agent_id=agent_id,
        thread_id=thread_id,
        permissions=permissions,
        timeout=timeout,
    ):
        chunk_count += 1
        if verbose:
            print(f"--- chunk {chunk_count} ---")
            print(json.dumps(chunk, indent=2, ensure_ascii=False))
            print()
        else:
            _print_chunk_summary(chunk)

        if chunk.get("kind") == "interrupt":
            awaiting_tool_permission = True
            action_requests = chunk.get("action_requests") or []

        status = stream_status_from_chunk(chunk)
        if status is not None:
            run_status = status
            if status == "awaiting_tool_permission":
                awaiting_tool_permission = True
                action_requests = chunk.get("action_requests") or []

        accumulated_text = accumulate_root_reply(accumulated_text, chunk)

    if awaiting_tool_permission:
        status = "awaiting_tool_permission"
    elif run_status:
        status = run_status
    else:
        status = "completed"
    return {
        "thread_id": thread_id,
        "agent_id": agent_id,
        "status": status,
        "run_status": run_status,
        "action_requests": action_requests,
        "accumulated_text": accumulated_text,
        "chunk_count": chunk_count,
    }


def _parse_args() -> tuple[str, str, int, bool]:
    """Parse CLI args: thread_id only, or base_url + thread_id."""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    argv = [arg for arg in sys.argv[1:] if arg not in ("--verbose", "-v")]

    if not argv:
        return DEFAULT_BASE_URL, DEFAULT_THREAD_ID, DEFAULT_MAX_ROUNDS, verbose
    if len(argv) == 1:
        arg = argv[0]
        if arg.startswith(("http://", "https://")):
            return arg.rstrip("/"), DEFAULT_THREAD_ID, DEFAULT_MAX_ROUNDS, verbose
        return DEFAULT_BASE_URL, arg, DEFAULT_MAX_ROUNDS, verbose

    max_rounds = DEFAULT_MAX_ROUNDS
    if len(argv) >= 3:
        try:
            max_rounds = int(argv[2])
        except ValueError:
            print(
                f"Invalid max_rounds={argv[2]!r}; using {DEFAULT_MAX_ROUNDS}.",
                file=sys.stderr,
            )
    return argv[0].rstrip("/"), argv[1], max_rounds, verbose


def main() -> int:
    """Run permit rounds over /stream until completed or max_rounds is reached."""
    base_url, thread_id, max_rounds, verbose = _parse_args()
    permissions = DEFAULT_PERMISSIONS

    if not thread_id:
        print(
            "thread_id is required. Pass it as argv[1] or set DEFAULT_THREAD_ID.",
            file=sys.stderr,
        )
        return 1
    if not permissions:
        print(
            "permissions is required. Set DEFAULT_PERMISSIONS in permit_stream.py.",
            file=sys.stderr,
        )
        return 1

    print(f"POST {base_url}/stream")
    print(f"agent_id={GENERAL_AGENT_ID}")
    print(f"thread_id={thread_id!r}")
    print(f"permissions={permissions!r}")
    print(f"max_rounds={max_rounds}")
    if verbose:
        print("verbose=True")
    print()

    result: dict[str, Any] | None = None
    try:
        for round_number in range(1, max_rounds + 1):
            print(f"--- permit round {round_number}/{max_rounds} ---")
            result = permit_stream_round(
                base_url=base_url,
                thread_id=thread_id,
                permissions=permissions,
                verbose=verbose,
            )
            status = result.get("status")
            action_requests = result.get("action_requests") or []

            if status == "completed":
                print(f"Status: completed (after {round_number} round(s)).")
                break

            if status != "awaiting_tool_permission":
                print(f"Status: {status!r} (stopping).")
                break

            print(f"Status: awaiting_tool_permission ({len(action_requests)} pending)")
            print(f"Pending:\n  {_summarize_action_requests(action_requests)}")

            if round_number == max_rounds:
                print(
                    f"Reached max_rounds={max_rounds}; still awaiting approval. "
                    "Run permit_stream.py again or increase max_rounds.",
                    file=sys.stderr,
                )
                break

            print("Approving and continuing...\n")
    except httpx.HTTPStatusError as exc:
        print(f"HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1
    except httpx.RequestError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        print("Is the API running? Try: python main.py", file=sys.stderr)
        return 1

    if result is None:
        return 1

    if result.get("chunk_count", 0) == 0:
        print("No chunks received.", file=sys.stderr)
        return 1

    _print_final_summary(
        thread_id=result.get("thread_id"),
        accumulated_text=str(result.get("accumulated_text", "")),
        chunk_count=result.get("chunk_count", 0),
        awaiting_tool_permission=result.get("status") == "awaiting_tool_permission",
        pending_action_count=len(result.get("action_requests") or []),
        run_status=result.get("run_status"),
    )

    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
