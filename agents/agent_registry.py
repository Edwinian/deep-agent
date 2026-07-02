"""Registry of agent specs keyed by agent ID."""

from agents.general_agent import GENERAL_AGENT, GENERAL_AGENT_ID
from agents.research_agent import RESEARCH_AGENT, RESEARCH_AGENT_ID
from agents.types import DeepAgent

AGENT_REGISTRY: dict[int, DeepAgent] = {
    RESEARCH_AGENT_ID: RESEARCH_AGENT,
    GENERAL_AGENT_ID: GENERAL_AGENT,
}
