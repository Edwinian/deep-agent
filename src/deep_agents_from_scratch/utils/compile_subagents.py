"""Compile DeepAgent specs for create_deep_agent subagents parameter."""

from __future__ import annotations

import json
from typing import cast

from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from langchain_core.language_models import BaseChatModel

from deep_agents_from_scratch.agents.constants import DEEP_AGENT_ONLY_FIELDS, DeepAgentField
from deep_agents_from_scratch.agents.types import DeepAgent, ModelConfig
from deep_agents_from_scratch.utils.compile_agent import compile_agent
from deep_agents_from_scratch.utils.resolve_model import resolve_model


def _to_subagent_spec(agent: DeepAgent) -> SubAgent:
    """Extract a SubAgent spec from a leaf DeepAgent."""
    spec = cast(
        SubAgent,
        {
            key: value
            for key, value in agent.items()
            if key not in DEEP_AGENT_ONLY_FIELDS
        },
    )
    if DeepAgentField.MODEL_CONFIG.value in agent:
        resolved = resolve_model(agent)
        if resolved is not None:
            spec["model"] = resolved
    if "tools" not in spec:
        spec["tools"] = []
    return spec


def _compile_subagent(
    agent: DeepAgent | CompiledSubAgent,
    default_model: str | BaseChatModel | None,
    default_model_config: ModelConfig | None = None,
) -> SubAgent | CompiledSubAgent:
    """Compile a single subagent spec, recursively handling nested subagents."""
    if "runnable" in agent:
        return agent

    nested_subagents = agent.get(DeepAgentField.SUBAGENTS.value)
    if not nested_subagents:
        return _to_subagent_spec(agent)

    runnable = compile_agent(
        agent,
        model=agent.get("model", default_model),
        model_config=agent.get(DeepAgentField.MODEL_CONFIG.value, default_model_config),
    )
    return CompiledSubAgent(
        name=agent["name"],
        description=json.dumps(
            {
                "description": agent["description"],
                "system_prompt": agent["system_prompt"],
            },
            ensure_ascii=False,
        ),
        runnable=runnable,
    )


def compile_subagents(
    subagents: list[DeepAgent | CompiledSubAgent],
    *,
    default_model: str | BaseChatModel | None = None,
    default_model_config: ModelConfig | None = None,
) -> list[SubAgent | CompiledSubAgent]:
    """Compile DeepAgent specs for create_deep_agent's subagents parameter."""
    return [
        _compile_subagent(agent, default_model, default_model_config)
        for agent in subagents
    ]
