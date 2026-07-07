"""Test client for POST /stream against the general agent."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Iterator
from typing import Any

import httpx

from agents.general_agent import GENERAL_AGENT_ID
from schemas.invoke_request import Permission
from utils.hitl import collect_action_requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MESSAGE = "What are the latest developments in LangGraph?"


def stream_agent(
    *,
    base_url: str = DEFAULT_BASE_URL,
    agent_id: int = GENERAL_AGENT_ID,
    message: str = DEFAULT_MESSAGE,
    thread_id: str | None = None,
    model_config: dict | None = None,
    permissions: list[Permission] | None = None,
    timeout: float = 300.0,
) -> Iterator[dict]:
    """Call the /stream endpoint and yield each NDJSON chunk."""
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
        json=payload,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            yield json.loads(line)


def _message_text(message: dict[str, Any]) -> str:
    """Return visible text from a serialized message."""
    data = message.get("data") or {}
    content = (data.get("content") or "").strip()
    if content:
        return content

    additional = data.get("additional_kwargs") or {}
    reasoning = (additional.get("reasoning_content") or "").strip()
    return reasoning


def _last_ai_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        msg_type = message.get("type")
        if msg_type == "ai" or (message.get("data") or {}).get("type") == "ai":
            return message
    return None


def _messages_from_values_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    messages = event.get("messages")
    return messages if isinstance(messages, list) else []


def _interrupts_from_values_event(event: dict[str, Any]) -> list[Any]:
    interrupts = event.get("__interrupt__")
    return interrupts if isinstance(interrupts, list) else []


def _summarize_update_event(event: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in event.items():
        if value is None:
            parts.append(key)
            continue
        if key == "model" and isinstance(value, dict):
            ai_message = _last_ai_message(_messages_from_values_event(value))
            if ai_message is None:
                parts.append("model")
                continue
            data = ai_message.get("data") or {}
            tool_calls = data.get("tool_calls") or []
            if tool_calls:
                tool_names = ", ".join(call.get("name", "?") for call in tool_calls)
                parts.append(f"model -> tool_calls: {tool_names}")
            else:
                text = _message_text(ai_message)
                preview = text.replace("\n", " ")[:80]
                parts.append(f"model -> {preview!r}" if preview else "model")
            continue
        if key == "tools" and isinstance(value, dict):
            tool_messages = _messages_from_values_event(value)
            if tool_messages:
                names = ", ".join(
                    (msg.get("data") or {}).get("name", "?") for msg in tool_messages
                )
                parts.append(f"tools: {names}")
            else:
                parts.append("tools")
            continue
        parts.append(key)
    return ", ".join(parts)


def _print_chunk_summary(chunk: dict[str, Any]) -> None:
    stream_mode = chunk.get("stream_mode", "?")
    graph = chunk.get("graph") or []
    graph_label = "/".join(graph) if graph else "root"
    event = chunk.get("event") or {}

    if stream_mode == "updates" and isinstance(event, dict):
        print(f"[{graph_label}] updates: {_summarize_update_event(event)}")
        return

    if stream_mode == "values" and isinstance(event, dict):
        messages = _messages_from_values_event(event)
        ai_message = _last_ai_message(messages)
        if ai_message is not None:
            text = _message_text(ai_message)
            preview = text.replace("\n", " ")[:80]
            if preview:
                print(f"[{graph_label}] values: last_ai={preview!r}")
            else:
                print(f"[{graph_label}] values: state updated ({len(messages)} messages)")
        else:
            print(f"[{graph_label}] values: state updated")


def _print_final_summary(
    *,
    thread_id: str | None,
    last_messages: list[dict[str, Any]],
    last_interrupts: list[Any],
    chunk_count: int,
) -> None:
    print()
    print("=" * 72)
    print("Stream summary")
    print("=" * 72)
    if thread_id:
        print(f"thread_id: {thread_id}")

    action_requests = collect_action_requests(last_interrupts) if last_interrupts else []
    if action_requests:
        print(f"status: awaiting_tool_permission ({len(action_requests)} pending)")
        for index, action_request in enumerate(action_requests, start=1):
            name = action_request["name"]
            args = action_request.get("args", {})
            query = args.get("query")
            if query is not None:
                print(f"  {index}. {name}: {query!r}")
            else:
                print(f"  {index}. {name}: {args!r}")
        print("Resume with: python permit.py", thread_id or "<thread_id>")
        print("         or: python permit_stream.py", thread_id or "<thread_id>")
        print("=" * 72)
        return

    ai_message = _last_ai_message(last_messages)
    print("status: completed")
    print()
    print("Agent response:")
    if ai_message is None:
        print("  (no AI message in final state)")
    else:
        text = _message_text(ai_message)
        data = ai_message.get("data") or {}
        content = (data.get("content") or "").strip()
        if content:
            print(content)
        elif text:
            print("  (no final text content; last model output was reasoning only)")
            print()
            print(text)
        else:
            print("  (empty — model stopped without a user-facing reply)")
            tool_calls = data.get("tool_calls") or []
            if tool_calls:
                names = ", ".join(call.get("name", "?") for call in tool_calls)
                print(f"  Pending tool calls: {names}")

    print()
    print(f"Received {chunk_count} chunk(s).")
    print("=" * 72)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream agent events from POST /stream.")
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
    args = _parse_args()
    base_url = args.base_url
    message = args.message

    print(f"POST {base_url}/stream")
    print(f"agent_id={GENERAL_AGENT_ID}")
    print(f"message={message!r}")
    print()

    thread_id: str | None = None
    chunk_count = 0
    last_messages: list[dict[str, Any]] = []
    last_interrupts: list[Any] = []

    try:
        for chunk in stream_agent(base_url=base_url, message=message):
            chunk_count += 1
            thread_id = chunk.get("thread_id", thread_id)

            if chunk_count == 1 and thread_id:
                print(f"thread_id={thread_id!r}")
                print()

            if args.verbose:
                print(f"--- chunk {chunk_count} ---")
                print(json.dumps(chunk, indent=2, ensure_ascii=False))
                print()
            else:
                _print_chunk_summary(chunk)

            event = chunk.get("event") or {}
            if chunk.get("stream_mode") == "values" and isinstance(event, dict):
                messages = _messages_from_values_event(event)
                if messages:
                    last_messages = messages
                interrupts = _interrupts_from_values_event(event)
                if interrupts:
                    last_interrupts = interrupts

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
        last_messages=last_messages,
        last_interrupts=last_interrupts,
        chunk_count=chunk_count,
    )
    return 0 if not last_interrupts else 1


if __name__ == "__main__":
    raise SystemExit(main())
