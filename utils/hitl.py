"""Human-in-the-loop helpers for invoke and resume flows."""

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException
from langchain.agents.middleware.human_in_the_loop import (
    Action,
    ApproveDecision,
    Decision,
    EditDecision,
    HITLResponse,
    RejectDecision,
    RespondDecision,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langgraph.types import Command, Interrupt as LangGraphInterrupt

from schemas.invoke_request import DecisionType, Permission


def is_resume_request(
    *,
    thread_id: str | None,
    permissions: list[Permission] | None,
) -> bool:
    """Return True when the client is resuming an interrupted thread."""
    return thread_id is not None and permissions is not None


def find_tool_call_id(action_request: dict[str, Any], messages: list[Any]) -> str | None:
    """Match an interrupt action request to a tool call id in message history."""
    for message in reversed(_iter_ai_messages(messages)):
        for tool_call in message.tool_calls:
            if (
                tool_call["name"] == action_request.get("name")
                and tool_call["args"] == action_request.get("args")
            ):
                return tool_call["id"]
    return None


def enrich_interrupt_tool_call_ids(
    serialized: dict[str, Any],
    messages: list[Any],
) -> None:
    """Attach tool_call_id to each action request in serialized interrupts."""
    interrupts = serialized.get("__interrupt__")
    if not interrupts:
        return

    for interrupt in interrupts:
        if isinstance(interrupt, LangGraphInterrupt):
            value = interrupt.value
        else:
            value = interrupt.get("value", {})

        if not isinstance(value, dict):
            continue

        for action_request in value.get("action_requests", []):
            tool_call_id = find_tool_call_id(action_request, messages)
            if tool_call_id is not None:
                action_request["tool_call_id"] = tool_call_id


def build_permission_message(interrupts: list[Any]) -> str:
    """Build a client-facing summary for pending tool approvals."""
    lines = ["Tool execution requires your approval before the agent can continue:"]

    for interrupt in interrupts:
        if isinstance(interrupt, LangGraphInterrupt):
            value = interrupt.value
        else:
            value = interrupt.get("value", {})

        if not isinstance(value, dict):
            continue

        for action_request in value.get("action_requests", []):
            name = action_request.get("name", "unknown_tool")
            description = action_request.get("description")
            tool_call_id = action_request.get("tool_call_id")
            suffix = f" (tool_call_id={tool_call_id})" if tool_call_id else ""
            if description:
                lines.append(f"- {name}{suffix}: {description}")
            else:
                lines.append(f"- {name}{suffix}: {action_request.get('args', {})}")

    lines.append(
        "Reply with the same thread_id and permissions to approve, edit, reject, or respond."
    )
    return "\n".join(lines)


def build_resume_command(
    interrupts: tuple[LangGraphInterrupt, ...] | list[Any],
    permissions: list[Permission],
    messages: list[Any],
) -> Command:
    """Build a LangGraph ``Command(resume=...)`` from client permissions.

    Mapping overview (client ``permissions`` → LangGraph resume payload):

    1. Index client permissions by ``tool_call_id`` (the id from the AI message).
    2. For each pending ``Interrupt`` from checkpoint state:
       - ``Interrupt.id`` is the LangGraph interrupt key (NOT a tool_call_id).
       - ``Interrupt.value.action_requests`` lists tools awaiting human review.
    3. Match each action request to its tool call in ``messages`` (see
       ``_match_tool_calls``) so we know which ``tool_call_id`` to look up.
    4. Convert each matched ``Permission`` into a HITL ``Decision`` (see
       ``_permission_to_decision``). Decisions must be in the same order as
       ``action_requests`` — LangGraph applies them positionally.
    5. Attach decisions under the interrupt id::
         {interrupt_id: {"decisions": [decision_0, decision_1, ...]}}
    6. Return ``Command(resume=...)`` — a single HITLResponse when there is one
       interrupt, otherwise a map keyed by interrupt id.
    """
    if not interrupts:
        raise HTTPException(
            status_code=400,
            detail="No interrupted tool calls found for this thread.",
        )

    # Client permissions are keyed by tool_call_id; each pending tool call must
    # supply exactly one permission entry before we can resume.
    permissions_by_id = {permission["tool_call_id"]: permission for permission in permissions}
    resume_payload: dict[str, HITLResponse] = {}

    for interrupt in interrupts:
        if isinstance(interrupt, LangGraphInterrupt):
            interrupt_id = interrupt.id
            value = interrupt.value
        else:
            interrupt_id = interrupt["id"]
            value = interrupt.get("value", {})

        action_requests = value.get("action_requests", [])
        # Resolve action_requests → concrete tool_calls from thread history.
        interrupted_tool_calls = _match_tool_calls(action_requests, messages)
        if len(interrupted_tool_calls) != len(action_requests):
            raise HTTPException(
                status_code=400,
                detail="Could not match all interrupted tool calls in thread history.",
            )

        decisions: list[Decision] = []
        for tool_call, action_request in zip(interrupted_tool_calls, action_requests):
            tool_call_id = tool_call["id"]
            # Look up the client permission for this pending tool call.
            permission = permissions_by_id.get(tool_call_id)
            if permission is None:
                tool_call_id = action_request.get("tool_call_id", tool_call_id)
                permission = permissions_by_id.get(tool_call_id)

            if permission is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing permission for tool_call_id={tool_call_id!r}.",
                )

            # Permission → HITL decision (approve / edit / reject / respond).
            decisions.append(_permission_to_decision(permission, tool_call))

        # LangGraph routes resume input by interrupt id, not tool_call_id.
        resume_payload[interrupt_id] = {"decisions": decisions}

    # Single interrupt: pass HITLResponse shape directly. Multiple: map by id.
    if len(resume_payload) == 1:
        return Command(resume=next(iter(resume_payload.values())))

    return Command(resume=resume_payload)


def _iter_ai_messages(messages: list[Any]) -> list[AIMessage]:
    """Normalize thread history into AIMessage objects that expose tool_calls."""
    ai_messages: list[AIMessage] = []
    for message in messages:
        if isinstance(message, AIMessage):
            ai_messages.append(message)
        elif isinstance(message, dict):
            data = message.get("data", message)
            if data.get("type") == "ai":
                tool_calls = data.get("tool_calls", [])
                ai_messages.append(
                    AIMessage(
                        content=data.get("content", ""),
                        tool_calls=tool_calls,
                        id=data.get("id"),
                    )
                )
    return ai_messages


def _match_tool_calls(
    action_requests: list[dict[str, Any]],
    messages: list[Any],
) -> list[ToolCall]:
    """Align each interrupt action_request with its tool call in message history.

    HumanInTheLoopMiddleware stores pending tools in ``action_requests`` without
    guaranteed tool_call_id fields. This function returns the corresponding
    ``tool_calls`` entries in the same order, which ``build_resume_command`` then
    uses to join client ``permissions`` (keyed by tool_call_id) to HITL decisions.
    """
    matched: list[ToolCall] = []
    seen_ids: set[str] = set()

    for action_request in action_requests:
        tool_call_id = action_request.get("tool_call_id")
        if tool_call_id:
            # Preferred path: action_request already carries tool_call_id from our
            # serialized invoke response.
            tool_call = _find_tool_call_by_id(tool_call_id, messages)
            if tool_call is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown tool_call_id={tool_call_id!r} in thread history.",
                )
            matched.append(tool_call)
            seen_ids.add(tool_call_id)
            continue

        # Fallback: locate the tool call by exact name + args match.
        for message in reversed(_iter_ai_messages(messages)):
            for tool_call in message.tool_calls:
                if tool_call["id"] in seen_ids:
                    continue
                if (
                    tool_call["name"] == action_request.get("name")
                    and tool_call["args"] == action_request.get("args")
                ):
                    matched.append(tool_call)
                    seen_ids.add(tool_call["id"])
                    break
            else:
                continue
            break
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not find tool call for interrupted action "
                    f"{action_request.get('name')!r}."
                ),
            )

    return matched


def _find_tool_call_by_id(tool_call_id: str, messages: list[Any]) -> ToolCall | None:
    """Return the tool call dict for ``tool_call_id`` from the newest AI message first."""
    for message in reversed(_iter_ai_messages(messages)):
        for tool_call in message.tool_calls:
            if tool_call["id"] == tool_call_id:
                return tool_call
    return None


def _permission_to_decision(
    permission: Permission,
    tool_call: ToolCall,
) -> Decision:
    """Convert one client Permission into one HumanInTheLoopMiddleware decision.

    This is the final step in the permissions → resume mapping. Each decision
    is consumed by LangGraph in the same order as the interrupted tool calls
    for that interrupt batch.
    """
    decision = permission["decision"]
    if isinstance(decision, str):
        decision = DecisionType(decision)

    if decision is DecisionType.APPROVE:
        return ApproveDecision(type=DecisionType.APPROVE)

    if decision is DecisionType.REJECT:
        reject_decision: RejectDecision = {"type": DecisionType.REJECT}
        reject_reason = permission.get("reject_reason") or permission.get("edit_instruction")
        if reject_reason:
            # Permission.reject_reason maps to RejectDecision.message.
            reject_decision["message"] = reject_reason
        return reject_decision

    if decision is DecisionType.RESPOND:
        message = permission.get("respond_instruction") or permission.get("edit_instruction")
        if not message:
            raise HTTPException(
                status_code=400,
                detail=f"respond_instruction is required for tool_call_id={permission['tool_call_id']!r}.",
            )
        return RespondDecision(type=DecisionType.RESPOND, message=message)

    if decision is DecisionType.EDIT:
        instruction = permission.get("edit_instruction")
        if not instruction:
            raise HTTPException(
                status_code=400,
                detail=f"edit_instruction is required for tool_call_id={permission['tool_call_id']!r}.",
            )
        edited_action: Action = {
            "name": tool_call["name"],
            "args": _parse_edit_instruction(instruction, tool_call["args"]),
        }
        return EditDecision(type=DecisionType.EDIT, edited_action=edited_action)

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported decision={decision!r} for tool_call_id={permission['tool_call_id']!r}.",
    )


def _parse_edit_instruction(
    instruction: str,
    original_args: Action["args"],
) -> Action["args"]:
    """Build edited tool args from ``edit_instruction``.

    Accepts either JSON args (full override) or plain text (mapped onto a
    common arg field like ``query`` for search tools).
    """
    try:
        parsed = json.loads(instruction)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return parsed

    updated_args = dict(original_args)
    if "query" in updated_args:
        updated_args["query"] = instruction
    elif "content" in updated_args:
        updated_args["content"] = instruction
    elif "description" in updated_args:
        updated_args["description"] = instruction
    else:
        updated_args["instruction"] = instruction
    return updated_args
