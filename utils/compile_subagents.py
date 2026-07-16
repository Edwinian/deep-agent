"""Compile DeepAgent specs for create_deep_agent subagents parameter."""

from __future__ import annotations

from typing import cast

from deepagents.middleware.subagents import CompiledSubAgent, SubAgent

from agents.constants import DEEP_AGENT_ONLY_FIELDS, DeepAgentField
from agents.types import DeepAgent, ModelConfig
from utils.compile_agent import compile_agent
from utils.resolve_model import resolve_model
from utils.task_tool_args_repair import ToolCallArgsRepairMiddleware


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
    # Leaf SubAgents skip compile_agent's DEFAULT_PII_MIDDLEWARE; inject
    # arg repair so HITL sees filled web_search_tool / task args.
    existing = list(spec.get("middleware") or [])
    if not any(isinstance(m, ToolCallArgsRepairMiddleware) for m in existing):
        existing.append(ToolCallArgsRepairMiddleware())
    spec["middleware"] = existing
    return spec


def _compile_subagent(
    agent: DeepAgent | CompiledSubAgent,
    default_model: str | None,
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
        description=agent["description"],
        runnable=runnable,
    )


def compile_subagents(
    subagents: list[DeepAgent | CompiledSubAgent],
    *,
    default_model: str | None = None,
    default_model_config: ModelConfig | None = None,
) -> list[SubAgent | CompiledSubAgent]:
    """Compile DeepAgent specs for create_deep_agent's subagents parameter."""
    return [
        _compile_subagent(agent, default_model, default_model_config)
        for agent in subagents
    ]
