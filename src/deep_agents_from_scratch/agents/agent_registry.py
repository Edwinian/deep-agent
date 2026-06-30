"""Registry of agent specs keyed by agent ID."""

from deep_agents_from_scratch.agents.general_agent import GENERAL_AGENT, GENERAL_AGENT_ID
from deep_agents_from_scratch.agents.research_agent import RESEARCH_AGENT, RESEARCH_AGENT_ID
from deep_agents_from_scratch.agents.types import DeepAgent

AGENT_REGISTRY: dict[int, DeepAgent] = {
    RESEARCH_AGENT_ID: RESEARCH_AGENT,
    GENERAL_AGENT_ID: GENERAL_AGENT,
}
