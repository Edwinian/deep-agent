"""Middleware to repair incomplete tool calls before HITL / execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from typing_extensions import override

from constants.agent_name import AgentName


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(part for part in parts if part).strip()
    return ""


def last_user_text(state_or_messages: Any) -> str:
    """Return the most recent human message text from state or a message list."""
    if isinstance(state_or_messages, list):
        messages = state_or_messages
    elif isinstance(state_or_messages, dict):
        messages = state_or_messages.get("messages") or []
    else:
        return ""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_text(message.content)
        if isinstance(message, dict) and message.get("type") == "human":
            data = message.get("data") or message
            content = data.get("content") if isinstance(data, dict) else None
            return _message_text(content)
    return ""


def repair_tool_call_args(
    tool_call: dict[str, Any],
    *,
    user_text: str,
    default_subagent_type: str = str(AgentName.RESEARCH_AGENT),
) -> dict[str, Any] | None:
    """Return a repaired tool_call dict, or None if unchanged."""
    name = tool_call.get("name")
    args = dict(tool_call.get("args") or {})
    changed = False

    if name == "task":
        if not str(args.get("subagent_type") or "").strip():
            args["subagent_type"] = default_subagent_type
            changed = True
        if not str(args.get("description") or "").strip() and user_text:
            args["description"] = (
                "Research and answer this user question with up-to-date "
                "sources. Return a concise factual answer with citations.\n\n"
                f"Question: {user_text}"
            )
            changed = True
    elif name == "web_search_tool":
        if not str(args.get("query") or "").strip() and user_text:
            args["query"] = user_text
            changed = True

    if not changed:
        return None
    return {**tool_call, "args": args}


def dedupe_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate tool calls with the same name and args."""
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for tool_call in tool_calls:
        name = str(tool_call.get("name") or "")
        args = tool_call.get("args") or {}
        key = (name, repr(sorted(args.items())) if isinstance(args, dict) else repr(args))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tool_call)
    return deduped


def _repair_ai_message(
    message: AIMessage,
    *,
    user_text: str,
    default_subagent_type: str,
) -> AIMessage | None:
    """Return a repaired AIMessage, or None if unchanged."""
    if not message.tool_calls:
        return None

    repaired_calls: list[dict[str, Any]] = []
    changed = False
    for tool_call in message.tool_calls:
        call_dict = dict(tool_call)
        repaired = repair_tool_call_args(
            call_dict,
            user_text=user_text,
            default_subagent_type=default_subagent_type,
        )
        if repaired is not None:
            repaired_calls.append(repaired)
            changed = True
        else:
            repaired_calls.append(call_dict)

    deduped_calls = dedupe_tool_calls(repaired_calls)
    if len(deduped_calls) != len(repaired_calls):
        changed = True

    if not changed:
        return None

    return AIMessage(
        content=message.content,
        id=message.id,
        name=message.name,
        tool_calls=deduped_calls,
        additional_kwargs=dict(message.additional_kwargs or {}),
        response_metadata=dict(getattr(message, "response_metadata", None) or {}),
    )


class ToolCallArgsRepairMiddleware(AgentMiddleware):
    """Fill missing tool args and dedupe identical calls before HITL.

    ``HumanInTheLoopMiddleware`` is appended *after* user middleware, and
    ``after_model`` runs last→first — so HITL would see unrepaired calls if we
    only patched in ``after_model``. Repair in ``wrap_model_call`` instead so
    the model node writes fixed tool_calls into state before HITL runs.
    """

    default_subagent_type: str = str(AgentName.RESEARCH_AGENT)

    def _user_text_from_request(self, request: ModelRequest[Any]) -> str:
        return last_user_text(request.messages) or last_user_text(request.state)

    def _repair_model_response_messages(
        self,
        request: ModelRequest[Any],
        result: list[Any],
    ) -> tuple[list[Any], bool]:
        user_text = self._user_text_from_request(request)
        new_result: list[Any] = []
        changed = False
        for message in result:
            if isinstance(message, AIMessage):
                repaired = _repair_ai_message(
                    message,
                    user_text=user_text,
                    default_subagent_type=self.default_subagent_type,
                )
                if repaired is not None:
                    new_result.append(repaired)
                    changed = True
                else:
                    new_result.append(message)
            else:
                new_result.append(message)
        return new_result, changed

    def _repair_model_response(
        self,
        request: ModelRequest[Any],
        response: ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        if isinstance(response, AIMessage):
            repaired = _repair_ai_message(
                response,
                user_text=self._user_text_from_request(request),
                default_subagent_type=self.default_subagent_type,
            )
            return repaired if repaired is not None else response

        if isinstance(response, ExtendedModelResponse):
            model_response = response.model_response
            new_result, changed = self._repair_model_response_messages(
                request,
                model_response.result,
            )
            if not changed:
                return response
            return ExtendedModelResponse(
                model_response=ModelResponse(
                    result=new_result,
                    structured_response=model_response.structured_response,
                ),
                command=response.command,
            )

        if not isinstance(response, ModelResponse):
            return response

        new_result, changed = self._repair_model_response_messages(
            request,
            response.result,
        )
        if not changed:
            return response
        return ModelResponse(
            result=new_result,
            structured_response=response.structured_response,
        )

    def _repair_request(self, request: ToolCallRequest) -> ToolCallRequest:
        repaired = repair_tool_call_args(
            dict(request.tool_call),
            user_text=last_user_text(request.state),
            default_subagent_type=self.default_subagent_type,
        )
        if repaired is None:
            return request
        return request.override(tool_call=repaired)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Any],
    ) -> Any:
        return self._repair_model_response(request, handler(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[Any]],
    ) -> Any:
        return self._repair_model_response(request, await handler(request))

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        return handler(self._repair_request(request))

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        return await handler(self._repair_request(request))


# Backward-compatible alias
TaskToolArgsRepairMiddleware = ToolCallArgsRepairMiddleware
