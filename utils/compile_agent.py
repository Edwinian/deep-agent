"""Compile a DeepAgent spec into a runnable deep agent graph."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from daytona import Daytona, DaytonaConfig
from daytona.common.daytona import CreateSandboxFromSnapshotParams
from daytona.common.errors import DaytonaNotFoundError
from daytona.common.sandbox import SandboxState
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage
from langchain_daytona import DaytonaSandbox
from langgraph.config import get_config
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer
from typing_extensions import override

from agents.types import DeepAgent, InterruptOn, ModelConfig
from tools.default_interrupt_on import DEFAULT_INTERRUPT_ON
from utils.get_checkpointer import CheckpointerType, get_checkpointer
from utils.resolve_model import resolve_model

DEFAULT_SANDBOX_AUTO_STOP_INTERVAL_SECONDS = 3600

logger = logging.getLogger(__name__)

_daytona_client_instance: Daytona | None = None

PII_GUARDRAIL_SYSTEM_APPENDIX = """\
Privacy guardrails:
- User messages may contain [REDACTED_*] placeholders where sensitive data was removed.
- Never treat a placeholder as real data or repeat it as an answer.
- If asked to recall, repeat, or confirm redacted information, explain that it was \
redacted for privacy and you cannot access or disclose it."""

REDACTED_PII_REFUSAL = (
    "I cannot access or disclose that information because it was redacted "
    "for privacy protection."
)
_REDACTED_ONLY_RESPONSE = re.compile(r"^\[REDACTED_[A-Z0-9_]+\]$")


class RedactedPIIResponseMiddleware(AgentMiddleware):
    """Replace placeholder-only AI replies and drop internal reasoning metadata."""

    @override
    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        messages = state["messages"]
        if not messages:
            return None

        last_ai_idx = None
        last_ai_msg = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                last_ai_msg = messages[i]
                last_ai_idx = i
                break

        if last_ai_idx is None or last_ai_msg is None:
            return None

        content = str(last_ai_msg.content or "").strip()
        additional_kwargs = dict(last_ai_msg.additional_kwargs or {})
        additional_kwargs.pop("reasoning_content", None)

        if _REDACTED_ONLY_RESPONSE.fullmatch(content):
            content = REDACTED_PII_REFUSAL

        if (
            content == str(last_ai_msg.content or "").strip()
            and additional_kwargs == (last_ai_msg.additional_kwargs or {})
        ):
            return None

        updated_message = AIMessage(
            content=content,
            id=last_ai_msg.id,
            name=last_ai_msg.name,
            tool_calls=last_ai_msg.tool_calls,
            additional_kwargs=additional_kwargs,
        )
        new_messages = list(messages)
        new_messages[last_ai_idx] = updated_message
        return {"messages": new_messages}


# Built-in PII types from LangChain guardrails, excluding url.
# https://docs.langchain.com/oss/python/langchain/guardrails#built-in-pii-types-and-configuration
DEFAULT_PII_MIDDLEWARE = tuple(
    PIIMiddleware(
        pii_type,
        apply_to_input=True,
        apply_to_output=True,
        apply_to_tool_results=True,
    )
    for pii_type in ("email", "credit_card", "ip", "mac_address", "url")
) + (RedactedPIIResponseMiddleware(),)


def _get_daytona_client() -> Daytona:
    """Return a process-wide Daytona client."""
    global _daytona_client_instance
    if _daytona_client_instance is None:
        api_key = os.getenv("DAYTONA_API_KEY")
        if not api_key:
            raise ValueError("DAYTONA_API_KEY is required for Daytona sandbox backend")
        _daytona_client_instance = Daytona(DaytonaConfig(api_key=api_key))
    return _daytona_client_instance


def _thread_id_from_runtime(runtime: ToolRuntime | Runtime) -> str:
    """Resolve thread_id from tool or model middleware runtime objects."""
    config = getattr(runtime, "config", None)
    if config is not None:
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            return str(thread_id)

    execution_info = getattr(runtime, "execution_info", None)
    if execution_info is not None and execution_info.thread_id:
        return str(execution_info.thread_id)

    thread_id = get_config().get("configurable", {}).get("thread_id")
    if thread_id:
        return str(thread_id)

    raise ValueError(
        "thread_id is required in config['configurable'] for sandbox backend"
    )


def _ensure_sandbox_started(client: Daytona, sandbox) -> None:
    """Start a stopped Daytona sandbox before use."""
    if sandbox.state == SandboxState.STARTED:
        return
    client.start(sandbox)


def _resolve_daytona_sandbox(thread_id: str):
    """Return an existing thread-scoped Daytona sandbox or create one."""
    sandbox_name = f"thread-{thread_id}"
    client = _get_daytona_client()

    try:
        sandbox = client.get(sandbox_name)
    except DaytonaNotFoundError:
        auto_stop_interval = int(
            os.getenv(
                "DAYTONA_SANDBOX_AUTO_STOP_INTERVAL_SECONDS",
                str(DEFAULT_SANDBOX_AUTO_STOP_INTERVAL_SECONDS),
            )
        )
        sandbox = client.create(
            CreateSandboxFromSnapshotParams(
                name=sandbox_name,
                auto_stop_interval=auto_stop_interval,
            )
        )
    else:
        _ensure_sandbox_started(client, sandbox)

    return sandbox


def get_sandbox(runtime: ToolRuntime | Runtime) -> BackendProtocol:
    """Resolve a thread-scoped Daytona sandbox backend for tool execution.

    Reuses an existing sandbox named ``thread-{thread_id}`` when present;
    otherwise creates one with an auto-stop interval for automatic cleanup.
    Falls back to ephemeral state storage when sandbox is disabled or unavailable.
    """
    if os.getenv("DAYTONA_SANDBOX_ENABLED", "true").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return StateBackend()

    thread_id = _thread_id_from_runtime(runtime)

    try:
        sandbox = _resolve_daytona_sandbox(thread_id)
        return DaytonaSandbox(sandbox=sandbox)
    except Exception as exc:
        logger.warning(
            "Daytona sandbox unavailable for thread %s; using StateBackend: %s",
            thread_id,
            exc,
        )
        return StateBackend()


def compile_agent(
    agent: DeepAgent,
    *,
    model: str | None = None,
    model_config: ModelConfig | None = None,
    checkpointer: Checkpointer | None = None,
    interrupt_on: InterruptOn | None = None,
) -> CompiledStateGraph:
    """Build a deep agent graph from a DeepAgent spec.

    Compiles nested subagents recursively, then calls create_deep_agent.
    Use this for the main orchestrator agent or any DeepAgent with subagents.
    """
    resolved_model = resolve_model(
        agent,
        model=model,
        model_config=model_config,
    )
    subagents = agent.get("subagents")
    if subagents:
        from utils.compile_subagents import compile_subagents

        compiled_subagents = compile_subagents(
            subagents,
            default_model=resolved_model,
            default_model_config=model_config,
        )
    else:
        compiled_subagents = None

    resolved_interrupt_on = (
        interrupt_on
        if interrupt_on is not None
        else agent.get("interrupt_on", DEFAULT_INTERRUPT_ON)
    )

    return create_deep_agent(
        tools=agent.get("tools"),
        system_prompt=f"{agent['system_prompt']}\n\n{PII_GUARDRAIL_SYSTEM_APPENDIX}",
        subagents=compiled_subagents,
        model=resolved_model,
        middleware=DEFAULT_PII_MIDDLEWARE,
        backend=get_sandbox,
        checkpointer=checkpointer or get_checkpointer(CheckpointerType.ASYNC_SQLITE),
        interrupt_on=resolved_interrupt_on,
    )
