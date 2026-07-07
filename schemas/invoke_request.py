"""Typed schemas for POST /invoke and POST /stream requests."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from agents.types import ModelConfig
from typing_extensions import NotRequired, TypedDict


class DecisionType(StrEnum):
    """Human decision for an interrupted tool call."""

    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"
    RESPOND = "respond"


class Permission(TypedDict):
    """Client-provided approval for pending tool calls, keyed by tool name."""

    name: str
    decision: DecisionType
    edit_instruction: NotRequired[str | None]
    respond_instruction: NotRequired[str | None]
    reject_reason: NotRequired[str | None]


class InvokeAgent(TypedDict):
    """Payload for POST /invoke and POST /stream."""

    agent_id: int
    thread_id: NotRequired[str | None]
    model_config: NotRequired[ModelConfig | None]
    message: NotRequired[str]
    permissions: NotRequired[list[Permission] | None]


def build_invoke_agent_from_stream_query(
    *,
    agent_id: int,
    thread_id: str | None = None,
    message: str | None = None,
    permissions: str | None = None,
    model_config: str | None = None,
) -> InvokeAgent:
    """Build an :class:`InvokeAgent` from GET /stream query parameters."""
    payload: dict[str, Any] = {"agent_id": agent_id}

    if thread_id is not None:
        payload["thread_id"] = thread_id
    if message is not None:
        payload["message"] = message

    if permissions is not None:
        try:
            parsed_permissions = json.loads(permissions)
        except json.JSONDecodeError as exc:
            raise ValueError("permissions must be valid JSON") from exc
        if not isinstance(parsed_permissions, list):
            raise ValueError("permissions must be a JSON array")
        payload["permissions"] = parsed_permissions

    if model_config is not None:
        try:
            parsed_model_config = json.loads(model_config)
        except json.JSONDecodeError as exc:
            raise ValueError("model_config must be valid JSON") from exc
        if not isinstance(parsed_model_config, dict):
            raise ValueError("model_config must be a JSON object")
        payload["model_config"] = parsed_model_config

    return payload  # type: ignore[typeddict-item]
