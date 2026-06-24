"""General orchestrator agent spec."""

from datetime import datetime

from deep_agents_from_scratch.agents.research_agent import RESEARCH_AGENT
from deep_agents_from_scratch.agents.types import DeepAgent
from deep_agents_from_scratch.prompts import (
    FILE_USAGE_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)

DEFAULT_MODEL = "xai:grok-3-mini"
MAX_CONCURRENT_RESEARCH_UNITS = 3
MAX_RESEARCHER_ITERATIONS = 3


def _build_subagent_instructions() -> str:
    return SUBAGENT_USAGE_INSTRUCTIONS.format(
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
        max_researcher_iterations=MAX_RESEARCHER_ITERATIONS,
        date=datetime.now().strftime("%a %b %-d, %Y"),
    )


def _build_system_prompt() -> str:
    subagent_instructions = _build_subagent_instructions()
    return (
        "# TODO MANAGEMENT\n"
        + TODO_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# FILE SYSTEM USAGE\n"
        + FILE_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# SUB-AGENT DELEGATION\n"
        + subagent_instructions
    )


GENERAL_AGENT: DeepAgent = {
    "name": "general-agent",
    "description": (
        "Orchestrates research by managing todos, files, and delegating to specialized agents."
    ),
    "system_prompt": _build_system_prompt(),
    "subagents": [RESEARCH_AGENT],
    "model": DEFAULT_MODEL,
}
