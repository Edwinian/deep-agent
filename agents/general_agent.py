"""General orchestrator agent spec."""

from datetime import datetime

from agents.research_agent import RESEARCH_AGENT, resolve_research_agent
from agents.types import DeepAgent
from constants.model_name import ModelName
from prompts import (
    FILE_USAGE_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)
from tools.get_math_mcp_tools import MATH_MCP_TOOLS_ID
from tools.get_weather_mcp_tools import WEATHER_MCP_TOOLS_ID

DEFAULT_MODEL = ModelName.GROK_4_3.with_provider()
MAX_CONCURRENT_RESEARCH_UNITS = 3
MAX_RESEARCHER_ITERATIONS = 3

GENERAL_AGENT_ID = 1002
GENERAL_AGENT_TOOL_IDS = [WEATHER_MCP_TOOLS_ID, MATH_MCP_TOOLS_ID]


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
    "id": GENERAL_AGENT_ID,
    "name": "general-agent",
    "description": (
        "Orchestrates research by managing todos, files, and delegating to specialized agents."
    ),
    "system_prompt": _build_system_prompt(),
    "subagents": [RESEARCH_AGENT],
    "model": DEFAULT_MODEL,
    "tools": [],
}


async def resolve_general_agent() -> DeepAgent:
    """Return the general agent spec with MCP tools loaded."""
    from tools.tool_registry import resolve_tools

    tools = await resolve_tools(GENERAL_AGENT_TOOL_IDS)
    research_agent = await resolve_research_agent()
    return {**GENERAL_AGENT, "tools": tools, "subagents": [research_agent]}
