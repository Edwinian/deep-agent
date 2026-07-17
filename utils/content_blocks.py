"""Parse and sanitize LangChain / model content-block payloads."""

from __future__ import annotations

import ast
import json
from typing import Any


def _try_parse_block_list(raw: str) -> list[Any] | None:
    """Parse a JSON or Python-literal list of content blocks, or return None."""
    stripped = raw.strip()
    if not stripped.startswith("["):
        return None

    try:
        parsed: Any = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError, MemoryError):
            return None

    if isinstance(parsed, list):
        return parsed
    return None


def _text_from_block_list(blocks: list[Any]) -> str | None:
    """Extract text from typed content blocks.

    Returns ``None`` when ``blocks`` does not look like a content-block list
    (so callers can fall back to the original string). Returns ``\"\"`` when the
    list is typed but has no user-facing text (e.g. tool_call-only dumps).
    """
    text_parts: list[str] = []
    saw_typed = False

    for block in blocks:
        if isinstance(block, str):
            text_parts.append(block)
            continue
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            saw_typed = True
            value = block.get("text")
            if value is None:
                value = block.get("content")
            if isinstance(value, str) and value:
                text_parts.append(value)
        elif block_type in {"reasoning", "thinking", "tool_call", "tool_use"}:
            saw_typed = True
        elif block_type is None and isinstance(block.get("text"), str):
            text_parts.append(str(block["text"]))

    if not saw_typed and not text_parts:
        return None
    if not saw_typed:
        return None
    return "".join(text_parts)


def is_non_text_content_block_dump(raw: str) -> bool:
    """True when ``raw`` is a content-block list with no user-facing text.

    Grok (and similar models) sometimes emit a stringified tool_call block on the
    text channel while also streaming a real tool_call chunk. Those dumps must not
    appear in the assistant reply.
    """
    blocks = _try_parse_block_list(raw)
    if blocks is None:
        return False
    extracted = _text_from_block_list(blocks)
    return extracted is not None and not extracted.strip()


def extract_user_text(content: Any) -> str:
    """Extract user-facing text from LangChain message content.

    Accepts plain strings, block lists, and stringified JSON/Python block lists.
    """
    if content is None:
        return ""

    if isinstance(content, list):
        extracted = _text_from_block_list(content)
        if extracted is not None:
            return extracted
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    if isinstance(content, str):
        blocks = _try_parse_block_list(content)
        if blocks is not None:
            extracted = _text_from_block_list(blocks)
            if extracted is not None:
                return extracted
        return content

    return ""
