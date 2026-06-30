"""Compile a DeepAgent spec into a runnable deep agent graph."""

from __future__ import annotations

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from deep_agents_from_scratch.agents.types import DeepAgent
from deep_agents_from_scratch.utils.compile_subagents import compile_subagents
from deep_agents_from_scratch.utils.get_checkpointer import CheckpointerType, get_checkpointer


def compile_agent(
    agent: DeepAgent,
    *,
    model: str | BaseChatModel | None = None,
    checkpointer: Checkpointer | None = None,
) -> CompiledStateGraph:
    """Build a deep agent graph from a DeepAgent spec.

    Compiles nested subagents recursively, then calls create_deep_agent.
    Use this for the main orchestrator agent or any DeepAgent with subagents.
    """
    resolved_model = agent.get("model", model)
    nested = agent.get("subagents")
    compiled_subagents = (
        compile_subagents(nested, model=resolved_model)
        if nested
        else None
    )

    return create_deep_agent(
        tools=agent.get("tools"),
        system_prompt=agent["system_prompt"],
        subagents=compiled_subagents,
        model=resolved_model,
        checkpointer=checkpointer or get_checkpointer(CheckpointerType.IN_MEMORY),
    )
