"""Agent definitions and factories."""

from deep_agents_from_scratch.agents.agent_registry import AGENT_REGISTRY
from deep_agents_from_scratch.agents.general_agent import (
    DEFAULT_MODEL,
    GENERAL_AGENT,
    GENERAL_AGENT_ID,
)
from deep_agents_from_scratch.agents.research_agent import RESEARCH_AGENT, RESEARCH_AGENT_ID
from deep_agents_from_scratch.agents.constants import DeepAgentField
from deep_agents_from_scratch.agents.types import DeepAgent, DeepAgentSubAgent, ModelConfig

__all__ = [
    "AGENT_REGISTRY",
    "DEFAULT_MODEL",
    "DeepAgent",
    "DeepAgentField",
    "DeepAgentSubAgent",
    "ModelConfig",
    "GENERAL_AGENT",
    "GENERAL_AGENT_ID",
    "RESEARCH_AGENT",
    "RESEARCH_AGENT_ID",
]
