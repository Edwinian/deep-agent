"""Middleware to repair incomplete tool calls before HITL / execution."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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


_USER_REQUEST_PATH = "/user_request.txt"


def _default_research_todos(user_text: str) -> list[dict[str, str]]:
    """Build a minimal research plan when the model omits write_todos args."""
    topic = " ".join(user_text.split()) if user_text else "the user request"
    return [
        {
            "content": f"Save the user request about {topic} to the filesystem",
            "status": "in_progress",
        },
        {
            "content": f"Research up-to-date information about {topic}",
            "status": "pending",
        },
        {
            "content": "Compile findings into a concise response",
            "status": "pending",
        },
    ]


def repair_tool_call_args(
    tool_call: dict[str, Any],
    *,
    user_text: str,
    default_subagent_type: str = str(AgentName.RESEARCH_AGENT),
) -> dict[str, Any] | None:
    """Return a repaired tool_call dict, or None if unchanged.

    Grok often emits the correct tool name with empty ``args``. Fill required
    fields with safe defaults so validation does not fail in a retry loop.
    """
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
                "sources. Return a concise factual answer only — do not "
                "include inline citations or a Sources section.\n\n"
                f"Question: {user_text}"
            )
            changed = True
    elif name == "web_search_tool":
        query = str(args.get("query") or "").strip()
        if not query and user_text:
            args["query"] = _search_query_from_text(user_text)
            changed = True
        elif query:
            repaired_query = _search_query_from_text(query)
            if repaired_query != query:
                args["query"] = repaired_query
                changed = True
    elif name == "ls":
        if not str(args.get("path") or "").strip():
            args["path"] = "/"
            changed = True
    elif name == "write_file":
        if not str(args.get("file_path") or "").strip():
            args["file_path"] = _USER_REQUEST_PATH
            changed = True
        if not str(args.get("content") or "").strip() and user_text:
            args["content"] = user_text
            changed = True
    elif name in {"read_file", "edit_file"}:
        if not str(args.get("file_path") or "").strip():
            args["file_path"] = _USER_REQUEST_PATH
            changed = True
    elif name == "write_todos":
        todos = args.get("todos")
        if not isinstance(todos, list) or not todos:
            args["todos"] = _default_research_todos(user_text)
            changed = True

    if not changed:
        return None
    return {**tool_call, "args": args}


def _search_query_from_text(text: str) -> str:
    """Build a concise web search query from a user or task brief."""
    match = re.search(r"(?im)^\s*question\s*:\s*(.+?)(?:\n|$)", text.strip())
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and any(
        "research and answer" in line.lower() or "up-to-date sources" in line.lower()
        for line in lines
    ):
        return re.sub(r"(?i)^question\s*:\s*", "", lines[-1]).strip()
    return " ".join(text.split())


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


def _tool_call_args_empty(tool_call: dict[str, Any]) -> bool:
    args = tool_call.get("args")
    if args is None:
        return True
    if isinstance(args, dict):
        return not any(str(value or "").strip() for value in args.values())
    if isinstance(args, str):
        return not args.strip()
    return False


def _has_successful_task_result(messages: list[Any] | None) -> bool:
    """True when a prior ``task`` tool already returned a non-empty answer."""
    if not messages:
        return False
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            if str(message.name or "") != "task":
                continue
            if _message_text(message.content):
                return True
            continue
        if isinstance(message, dict) and message.get("type") == "tool":
            data = message.get("data") or message
            if not isinstance(data, dict):
                continue
            if str(data.get("name") or message.get("name") or "") != "task":
                continue
            if _message_text(data.get("content")):
                return True
    return False


def _repair_ai_message(
    message: AIMessage,
    *,
    user_text: str,
    default_subagent_type: str,
    prior_messages: list[Any] | None = None,
) -> AIMessage | None:
    """Return a repaired AIMessage, or None if unchanged."""
    if not message.tool_calls:
        return None

    drop_empty_task = _has_successful_task_result(prior_messages)
    repaired_calls: list[dict[str, Any]] = []
    changed = False
    for tool_call in message.tool_calls:
        call_dict = dict(tool_call)
        name = str(call_dict.get("name") or "")
        # Grok often re-emits an empty ``task`` after research already succeeded.
        # Auto-filling that re-runs the same search and re-triggers HITL.
        if name == "task" and _tool_call_args_empty(call_dict) and drop_empty_task:
            changed = True
            continue
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
        prior_messages = list(request.messages or [])
        new_result: list[Any] = []
        changed = False
        for message in result:
            if isinstance(message, AIMessage):
                repaired = _repair_ai_message(
                    message,
                    user_text=user_text,
                    default_subagent_type=self.default_subagent_type,
                    prior_messages=prior_messages,
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
                prior_messages=list(request.messages or []),
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
