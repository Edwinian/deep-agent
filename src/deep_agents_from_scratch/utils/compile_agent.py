"""Compile a DeepAgent spec into a runnable deep agent graph."""

from __future__ import annotations

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from deep_agents_from_scratch.agents.types import DeepAgent, ModelConfig
from deep_agents_from_scratch.utils.get_checkpointer import CheckpointerType, get_checkpointer
from deep_agents_from_scratch.utils.resolve_model import resolve_model
from deep_agents_from_scratch.utils.compile_subagents import compile_subagents

def compile_agent(
    agent: DeepAgent,
    *,
    model: str | BaseChatModel | None = None,
    model_config: ModelConfig | None = None,
    checkpointer: Checkpointer | None = None,
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

        compiled_subagents = compile_subagents(
            subagents,
            default_model=resolved_model,
            default_model_config=model_config,
        )
    else:
        compiled_subagents = None

    return create_deep_agent(
        tools=agent.get("tools"),
        system_prompt=agent["system_prompt"],
        subagents=compiled_subagents,
        model=resolved_model,
        checkpointer=checkpointer or get_checkpointer(CheckpointerType.IN_MEMORY),
    )
