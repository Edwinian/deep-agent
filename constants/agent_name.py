"""Agent and agent system-prompt name constants."""

from enum import StrEnum


class AgentName(StrEnum):
    """Names for agents and their SystemPrompt rows."""

    RESEARCH_AGENT = "research_agent"
    GENERAL_AGENT = "general_agent"
    RAG_AGENT = "rag_agent"
