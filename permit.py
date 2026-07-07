"""Test client for resuming interrupted tool calls via POST /invoke."""

from __future__ import annotations

import json
import sys

import httpx

from agents.general_agent import GENERAL_AGENT_ID
from schemas.invoke_request import DecisionType, Permission

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


def permit_agent(
    *,
    thread_id: str,
    permissions: list[Permission],
    agent_id: int = GENERAL_AGENT_ID,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 300.0,
) -> dict:
    """Resume an interrupted thread by approving, editing, rejecting, or responding to tool calls."""
    payload = {
        "agent_id": agent_id,
        "thread_id": thread_id,
        "permissions": permissions,
    }

    response = httpx.post(
        f"{base_url.rstrip('/')}/invoke",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _parse_args() -> tuple[str, str, int]:
    """Parse CLI args: thread_id only, or base_url + thread_id."""
    if len(sys.argv) <= 1:
        return DEFAULT_BASE_URL, DEFAULT_THREAD_ID, DEFAULT_MAX_ROUNDS
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg.startswith(("http://", "https://")):
            return arg.rstrip("/"), DEFAULT_THREAD_ID, DEFAULT_MAX_ROUNDS
        return DEFAULT_BASE_URL, arg, DEFAULT_MAX_ROUNDS
    max_rounds = DEFAULT_MAX_ROUNDS
    if len(sys.argv) >= 4:
        try:
            max_rounds = int(sys.argv[3])
        except ValueError:
            print(f"Invalid max_rounds={sys.argv[3]!r}; using {DEFAULT_MAX_ROUNDS}.", file=sys.stderr)
    return sys.argv[1].rstrip("/"), sys.argv[2], max_rounds


def main() -> int:
    base_url, thread_id, max_rounds = _parse_args()
    permissions = DEFAULT_PERMISSIONS

    if not thread_id:
        print(
            "thread_id is required. Pass it as argv[1] or set DEFAULT_THREAD_ID.",
            file=sys.stderr,
        )
        return 1
    if not permissions:
        print("permissions is required. Set DEFAULT_PERMISSIONS in permit.py.", file=sys.stderr)
        return 1

    print(f"POST {base_url}/invoke")
    print(f"agent_id={GENERAL_AGENT_ID}")
    print(f"thread_id={thread_id!r}")
    print(f"permissions={permissions!r}")
    print(f"max_rounds={max_rounds}")
    print()

    result: dict | None = None
    try:
        for round_number in range(1, max_rounds + 1):
            print(f"--- permit round {round_number}/{max_rounds} ---")
            result = permit_agent(
                base_url=base_url,
                thread_id=thread_id,
                permissions=permissions,
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
                    "Run permit.py again or increase max_rounds.",
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

    print()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
