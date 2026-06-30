"""Compile DeepAgent specs for create_deep_agent subagents parameter."""

from __future__ import annotations

import json

from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from langchain_core.language_models import BaseChatModel

from deep_agents_from_scratch.agents.types import DeepAgent


def _to_subagent_spec(agent: DeepAgent) -> SubAgent:
    """Extract a SubAgent spec from a leaf DeepAgent."""
    spec: SubAgent = {
        "name": agent["name"],
        "description": agent["description"],
        "system_prompt": agent["system_prompt"],
        "tools": agent.get("tools") or [],
    }
    if "model" in agent:
        spec["model"] = agent["model"]
    if "middleware" in agent:
        spec["middleware"] = agent["middleware"]
    if "interrupt_on" in agent:
        spec["interrupt_on"] = agent["interrupt_on"]
    return spec


def _compile_subagent(
    agent: DeepAgent | CompiledSubAgent,
    default_model: str | BaseChatModel | None,
) -> SubAgent | CompiledSubAgent:
    """Compile a single subagent spec, recursively handling nested subagents."""
    if "runnable" in agent:
        return agent

    nested = agent.get("subagents")
    if not nested:
        return _to_subagent_spec(agent)

    from deep_agents_from_scratch.utils.compile_agent import compile_agent

    model = agent.get("model", default_model)
    runnable = compile_agent(agent, model=model)
    return {
        "name": agent["name"],
        "description": json.dumps(
            {
                "description": agent["description"],
                "system_prompt": agent["system_prompt"],
            },
            ensure_ascii=False,
        ),
        "runnable": runnable,
    }


def compile_subagents(
    subagents: list[DeepAgent | CompiledSubAgent],
    *,
    model: str | BaseChatModel | None = None,
) -> list[SubAgent | CompiledSubAgent]:
    """Compile DeepAgent specs for create_deep_agent's subagents parameter."""
    return [_compile_subagent(agent, model) for agent in subagents]
