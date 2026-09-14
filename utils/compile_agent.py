"""Compile a DeepAgent spec into a runnable deep agent graph."""

from __future__ import annotations

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgentMiddleware
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from agents.types import DeepAgent, InterruptOn, ModelConfig
from guardrails import GUARDRAILS
from prompts.pii_guardrails import PII_GUARDRAILS
from prompts.task_description_prefix import TASK_TOOL_DESCRIPTION
from tools.default_interrupt_on import DEFAULT_INTERRUPT_ON
from utils.daytona_sandbox import filesystem_backend
from utils.get_checkpointer import CheckpointerType, get_checkpointer
from utils.resolve_model import resolve_model

PII_GUARDRAIL_SYSTEM_APPENDIX = PII_GUARDRAILS


def _patch_task_tool_description() -> None:
    """Use a shorter task tool description so models keep required args."""
    if getattr(SubAgentMiddleware.__init__, "_deep_agents_task_desc_patched", False):
        return

    original_init = SubAgentMiddleware.__init__

    def patched_init(self, *args, task_description=None, **kwargs):  # type: ignore[no-untyped-def]
        if task_description is None:
            task_description = TASK_TOOL_DESCRIPTION
        original_init(self, *args, task_description=task_description, **kwargs)

    patched_init._deep_agents_task_desc_patched = True  # type: ignore[attr-defined]
    SubAgentMiddleware.__init__ = patched_init  # type: ignore[method-assign]


_patch_task_tool_description()


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

    skill_paths = agent.get("skill_paths")

    return create_deep_agent(
        tools=agent.get("tools"),
        system_prompt=f"{agent['system_prompt']}\n\n{PII_GUARDRAIL_SYSTEM_APPENDIX}",
        subagents=compiled_subagents,
        model=resolved_model,
        middleware=GUARDRAILS,
        # Pass backend instance or StateBackend factory for deepagents 0.3.x.
        backend=filesystem_backend(),
        checkpointer=checkpointer or get_checkpointer(CheckpointerType.ASYNC_SQLITE),
        interrupt_on=resolved_interrupt_on,
        skills=skill_paths,
    )
