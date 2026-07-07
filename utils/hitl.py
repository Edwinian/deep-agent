"""Human-in-the-loop helpers for invoke and resume flows."""

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException
from langchain.agents.middleware.human_in_the_loop import (
    Action,
    ActionRequest,
    ApproveDecision,
    Decision,
    EditDecision,
    HITLResponse,
    RejectDecision,
    RespondDecision,
)
from langchain_core.messages import ToolCall
from langgraph.types import Command, Interrupt as LangGraphInterrupt, StateSnapshot

from schemas.invoke_request import DecisionType, Permission


def collect_pending_interrupts(snapshot: StateSnapshot) -> tuple[LangGraphInterrupt, ...]:
    """Collect pending interrupts from the root graph and nested subgraph tasks."""
    collected: list[LangGraphInterrupt] = list(snapshot.interrupts)
    seen_ids: set[str] = {interrupt.id for interrupt in collected}

    def walk(state: StateSnapshot) -> None:
        for task in state.tasks:
            for interrupt in task.interrupts:
                if interrupt.id not in seen_ids:
                    seen_ids.add(interrupt.id)
                    collected.append(interrupt)
            if isinstance(task.state, StateSnapshot):
                walk(task.state)

    walk(snapshot)
    return tuple(collected)


def is_resume_request(
    *,
    thread_id: str | None,
    permissions: list[Permission] | None,
) -> bool:
    """Return True when the client is resuming an interrupted thread."""
    return thread_id is not None and permissions is not None


def _interrupt_value(interrupt: LangGraphInterrupt | dict[str, Any]) -> dict[str, Any]:
    if isinstance(interrupt, LangGraphInterrupt):
        value = interrupt.value
    else:
        value = interrupt.get("value", {})
    return value if isinstance(value, dict) else {}


def collect_action_requests(
    interrupts: tuple[LangGraphInterrupt, ...] | list[Any],
) -> list[ActionRequest]:
    """Flatten LangGraph interrupt payloads into client-facing action requests."""
    action_requests: list[ActionRequest] = []

    for interrupt in interrupts:
        value = _interrupt_value(interrupt)
        for action_request in value.get("action_requests", []):
            item: ActionRequest = {
                "name": action_request["name"],
                "args": action_request["args"],
            }
            description = action_request.get("description")
            if description is not None:
                item["description"] = description
            action_requests.append(item)

    return action_requests


def build_resume_command(
    interrupts: tuple[LangGraphInterrupt, ...] | list[Any],
    permissions: list[Permission],
) -> Command:
    """Build a LangGraph ``Command(resume=...)`` from client permissions.

    Client permissions are keyed by tool ``name``. One permission entry applies
    to every pending ``action_request`` with that name. LangGraph consumes the
    resulting ``decisions`` list in the same order as ``action_requests``.
    """
    if not interrupts:
        raise HTTPException(
            status_code=400,
            detail="No interrupted tool calls found for this thread.",
        )

    permissions_by_name = {permission["name"]: permission for permission in permissions}
    resume_payload: dict[str, HITLResponse] = {}

    for interrupt in interrupts:
        if isinstance(interrupt, LangGraphInterrupt):
            interrupt_id = interrupt.id
        else:
            interrupt_id = interrupt["id"]

        action_requests = _interrupt_value(interrupt).get("action_requests", [])
        required_names = {action_request["name"] for action_request in action_requests}
        missing_names = required_names - permissions_by_name.keys()
        if missing_names:
            missing = ", ".join(sorted(missing_names))
            raise HTTPException(
                status_code=400,
                detail=f"Missing permission for tool name(s): {missing}.",
            )

        decisions: list[Decision] = []
        for action_request in action_requests:
            permission = permissions_by_name[action_request["name"]]
            decisions.append(
                _permission_to_decision(
                    permission,
                    _action_request_as_tool_call(action_request),
                )
            )

        resume_payload[interrupt_id] = {"decisions": decisions}

    # Always key resume values by interrupt id. Subgraph interrupts (e.g. research
    # subagent inside task) may not receive a bare HITLResponse reliably.
    return Command(resume=resume_payload)


def _action_request_as_tool_call(action_request: dict[str, Any]) -> ToolCall:
    """Build a minimal tool call shape for decision conversion."""
    return ToolCall(
        type="tool_call",
        name=action_request["name"],
        args=action_request["args"],
        id="",
    )


def _permission_to_decision(
    permission: Permission,
    tool_call: ToolCall,
) -> Decision:
    """Convert one client Permission into one HumanInTheLoopMiddleware decision."""
    decision = permission["decision"]
    if isinstance(decision, str):
        decision = DecisionType(decision)

    if decision is DecisionType.APPROVE:
        return ApproveDecision(type=DecisionType.APPROVE)

    if decision is DecisionType.REJECT:
        reject_decision: RejectDecision = {"type": DecisionType.REJECT}
        reject_reason = permission.get("reject_reason") or permission.get("edit_instruction")
        if reject_reason:
            reject_decision["message"] = reject_reason
        return reject_decision

    if decision is DecisionType.RESPOND:
        message = permission.get("respond_instruction") or permission.get("edit_instruction")
        if not message:
            raise HTTPException(
                status_code=400,
                detail=f"respond_instruction is required for tool {permission['name']!r}.",
            )
        return RespondDecision(type=DecisionType.RESPOND, message=message)

    if decision is DecisionType.EDIT:
        instruction = permission.get("edit_instruction")
        if not instruction:
            raise HTTPException(
                status_code=400,
                detail=f"edit_instruction is required for tool {permission['name']!r}.",
            )
        edited_action: Action = {
            "name": tool_call["name"],
            "args": _parse_edit_instruction(instruction, tool_call["args"]),
        }
        return EditDecision(type=DecisionType.EDIT, edited_action=edited_action)

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported decision={decision!r} for tool {permission['name']!r}.",
    )


def _parse_edit_instruction(
    instruction: str,
    original_args: Action["args"],
) -> Action["args"]:
    """Build edited tool args from ``edit_instruction``."""
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
