"""Agent definitions and factories."""

from agents.agent_registry import AGENT_REGISTRY
from agents.general_agent import (
    DEFAULT_MODEL,
    GENERAL_AGENT,
    GENERAL_AGENT_ID,
)
from agents.research_agent import RESEARCH_AGENT, RESEARCH_AGENT_ID
from agents.constants import DeepAgentField
from agents.types import DeepAgent, DeepAgentSubAgent, ModelConfig

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
