"""Test client for POST /stream against the general agent."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Iterator

import httpx

from agents.ids import GENERAL_AGENT_ID

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MESSAGE = "What are the latest developments in LangGraph?"


def _json_payload(payload: dict) -> dict:
    """Return a JSON-serializable copy of a stream request payload."""
    return json.loads(
        json.dumps(
            payload,
            default=lambda item: item.value if hasattr(item, "value") else item,
        )
    )


def iter_sse_data_events(lines: Iterator[str]) -> Iterator[dict]:
    """Parse ``data:`` lines from a Server-Sent Events response body."""
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].lstrip()
        if not payload:
            continue
        yield json.loads(payload)


def stream_agent(
    *,
    base_url: str = DEFAULT_BASE_URL,
    agent_id: int = GENERAL_AGENT_ID,
    message: str = DEFAULT_MESSAGE,
    thread_id: str | None = None,
    model_config: dict | None = None,
    permissions: list[dict] | None = None,
    timeout: float = 300.0,
) -> Iterator[dict]:
    """Call the POST /stream SSE endpoint and yield each event payload."""
    resolved_thread_id = thread_id or str(uuid.uuid4())
    payload: dict = {
        "agent_id": agent_id,
        "thread_id": resolved_thread_id,
    }
    if model_config is not None:
        payload["model_config"] = model_config
    if permissions is not None:
        payload["permissions"] = permissions
    else:
        payload["message"] = message

    with httpx.stream(
        "POST",
        f"{base_url.rstrip('/')}/stream",
        json=_json_payload(payload),
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        yield from iter_sse_data_events(response.iter_lines())


class _StreamPrinter:
    """Stateful printer that mirrors the event-streaming doc output."""

    def __init__(self) -> None:
        self._subagent_prefix_printed = False

    def print_chunk(self, chunk: dict) -> None:
        kind = chunk.get("kind", "text")
        source = chunk.get("source", "agent")

        if kind == "subagent_started":
            self._subagent_prefix_printed = False
            subagent_name = chunk.get("subagent_name")
            if subagent_name:
                print(f"{subagent_name}: ", end="", flush=True)
                self._subagent_prefix_printed = True
            return

        if kind == "subagent_finished":
            print()
            self._subagent_prefix_printed = False
            return

        if source != "subagent":
            self._subagent_prefix_printed = False

        if kind == "reasoning":
            delta = chunk.get("delta", "")
            if delta:
                prefix = self._subagent_reasoning_prefix(chunk)
                print(f"{prefix}[thinking] {delta}", end="", flush=True)
            return

        if kind == "text":
            delta = chunk.get("delta", "")
            if delta:
                self._maybe_print_subagent_prefix(chunk)
                print(delta, end="", flush=True)
            return

        if kind == "message_tool_call_chunk":
            prefix = self._subagent_label(chunk)
            print(f"{prefix}tool call chunk: {chunk.get('tool_call_chunk')}")
            return

        if kind == "message_tool_calls_finalized":
            tool_calls = chunk.get("tool_calls")
            if tool_calls:
                prefix = self._subagent_label(chunk)
                print(f"{prefix}finalized tool calls: {tool_calls}")
            return

        if kind == "tool_call_started":
            tool_name = chunk.get("tool_name", "unknown")
            tool_input = chunk.get("input", {})
            prefix = self._subagent_label(chunk)
            print(f"{prefix}{tool_name}({tool_input})")
            return

        if kind == "tool_call_output_delta":
            delta = chunk.get("delta", "")
            if delta:
                self._maybe_print_subagent_prefix(chunk)
                print(delta, end="", flush=True)
            return

        if kind == "tool_call_finished":
            print()
            output = chunk.get("output")
            error = chunk.get("error")
            prefix = self._subagent_label(chunk)
            if output is not None:
                print(f"{prefix}{output}", end=" ")
            if error:
                print(f"{prefix}{error}", end="")
            print()
            return

        if kind == "interrupt":
            prefix = self._subagent_label(chunk)
            action_requests = chunk.get("action_requests") or []
            interrupt_ids = chunk.get("interrupt_ids") or []
            print()
            print(f"{prefix}Awaiting tool approval ({len(action_requests)} action(s))")
            if interrupt_ids:
                print(f"{prefix}interrupt_ids: {interrupt_ids}")
            for index, action in enumerate(action_requests, start=1):
                name = action.get("name", "unknown")
                args = action.get("args", {})
                print(f"{prefix}  {index}. {name}: {args!r}")
            print()
            return

        if kind == "message_finished":
            return

        if kind == "run_finished":
            return

    def _maybe_print_subagent_prefix(self, chunk: dict) -> None:
        if chunk.get("source") != "subagent" or self._subagent_prefix_printed:
            return
        subagent_name = chunk.get("subagent_name")
        if subagent_name:
            print(f"{subagent_name}: ", end="", flush=True)
            self._subagent_prefix_printed = True

    def _subagent_label(self, chunk: dict) -> str:
        if chunk.get("source") != "subagent":
            return ""
        subagent_name = chunk.get("subagent_name")
        if not subagent_name:
            return "[subagent] "
        return f"[{subagent_name}] "

    def _subagent_reasoning_prefix(self, chunk: dict) -> str:
        if chunk.get("source") != "subagent":
            return ""
        return self._subagent_label(chunk)


def _print_chunk_summary(chunk: dict) -> None:
    """Print one streamed chunk, matching the event-streaming doc output."""
    _STREAM_PRINTER.print_chunk(chunk)


_STREAM_PRINTER = _StreamPrinter()


def accumulate_root_reply(accumulated_text: str, chunk: dict) -> str:
    """Accumulate the root agent reply from streamed chunks."""
    if chunk.get("source", "agent") != "agent":
        return accumulated_text

    kind = chunk.get("kind")
    if kind == "text":
        return accumulated_text + str(chunk.get("delta", ""))

    if kind == "message_finished":
        content = chunk.get("content")
        if isinstance(content, str) and content:
            if not accumulated_text:
                return content
            if content.startswith(accumulated_text):
                return content
        return accumulated_text

    if kind == "run_finished":
        reply = chunk.get("reply")
        if isinstance(reply, str) and reply:
            return reply

    return accumulated_text


def stream_status_from_chunk(chunk: dict) -> str | None:
    """Return invoke-equivalent status when a run_finished chunk arrives."""
    if chunk.get("kind") != "run_finished":
        return None
    status = chunk.get("status")
    return status if isinstance(status, str) else None


def _print_final_summary(
    *,
    thread_id: str | None,
    accumulated_text: str,
    chunk_count: int,
    awaiting_tool_permission: bool = False,
    pending_action_count: int = 0,
    run_status: str | None = None,
) -> None:
    """Print end-of-run summary after text streaming completes."""
    print()
    print()
    print("=" * 72)
    print("Stream summary")
    print("=" * 72)
    if thread_id:
        print(f"thread_id: {thread_id}")
    if awaiting_tool_permission:
        print("status: awaiting_tool_permission")
        print(f"pending actions: {pending_action_count}")
        print()
        print("Resume with permit_stream.py and this thread_id.")
    else:
        print(f"status: {run_status or 'completed'}")
        print()
        print("Agent response:")
        if accumulated_text.strip():
            print(accumulated_text.strip())
        else:
            print("  (no text deltas received)")
    print()
    print(f"Received {chunk_count} chunk(s).")
    print("=" * 72)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the stream test client."""
    parser = argparse.ArgumentParser(description="Stream agent text from POST /stream.")
    parser.add_argument("base_url", nargs="?", default=DEFAULT_BASE_URL)
    parser.add_argument("message", nargs="?", default=DEFAULT_MESSAGE)
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full JSON for every chunk.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the stream test client and print streamed text."""
    args = _parse_args()
    base_url = args.base_url
    message = args.message

    print(f"POST {base_url}/stream")
    print(f"agent_id={GENERAL_AGENT_ID}")
    print(f"message={message!r}")
    print()

    thread_id: str | None = None
    chunk_count = 0
    accumulated_text = ""
    awaiting_tool_permission = False
    pending_action_count = 0
    run_status: str | None = None

    try:
        for chunk in stream_agent(base_url=base_url, message=message):
            chunk_count += 1
            thread_id = chunk.get("thread_id", thread_id)

            if chunk_count == 1 and thread_id:
                print(f"thread_id={thread_id!r}")
                print()

            if chunk.get("kind") == "interrupt":
                awaiting_tool_permission = True
                pending_action_count = len(chunk.get("action_requests") or [])

            status = stream_status_from_chunk(chunk)
            if status is not None:
                run_status = status
                if status == "awaiting_tool_permission":
                    awaiting_tool_permission = True
                    pending_action_count = len(chunk.get("action_requests") or [])

            if args.verbose:
                print(f"--- chunk {chunk_count} ---")
                print(json.dumps(chunk, indent=2, ensure_ascii=False))
                print()
            else:
                _print_chunk_summary(chunk)

            accumulated_text = accumulate_root_reply(accumulated_text, chunk)

    except httpx.HTTPStatusError as exc:
        print(f"HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1
    except httpx.RequestError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        print("Is the API running? Try: python main.py", file=sys.stderr)
        return 1

    if chunk_count == 0:
        print("No chunks received.", file=sys.stderr)
        return 1

    _print_final_summary(
        thread_id=thread_id,
        accumulated_text=accumulated_text,
        chunk_count=chunk_count,
        awaiting_tool_permission=awaiting_tool_permission,
        pending_action_count=pending_action_count,
        run_status=run_status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
