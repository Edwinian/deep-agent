"""Agent definitions and factories."""

from deep_agents_from_scratch.agents.general_agent import DEFAULT_MODEL, GENERAL_AGENT
from deep_agents_from_scratch.agents.research_agent import RESEARCH_AGENT
from deep_agents_from_scratch.agents.types import DeepAgent, DeepAgentSubAgent

__all__ = [
    "DEFAULT_MODEL",
    "DeepAgent",
    "DeepAgentSubAgent",
    "GENERAL_AGENT",
    "RESEARCH_AGENT",
]
