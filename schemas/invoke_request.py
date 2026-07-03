"""Typed schemas for POST /invoke and POST /stream requests."""

from __future__ import annotations

from enum import StrEnum

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
