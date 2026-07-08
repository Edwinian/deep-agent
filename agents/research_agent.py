"""Research agent spec."""

from agents.types import DeepAgent
from prompts import RESEARCHER_INSTRUCTIONS
from tools.think_tool import THINK_TOOL_ID
from tools.web_search_tool import WEB_SEARCH_TOOL_ID, get_today_str

RESEARCH_AGENT_ID = 1001
RESEARCH_AGENT_TOOL_IDS = [WEB_SEARCH_TOOL_ID, THINK_TOOL_ID]

RESEARCH_AGENT: DeepAgent = {
    "id": RESEARCH_AGENT_ID,
    "name": "research-agent",
    "description": (
        "Delegate research to the sub-agent researcher. "
        "Only give this researcher one topic at a time."
    ),
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    "tools": [],
}


async def resolve_research_agent() -> DeepAgent:
    """Return the research agent spec with tools loaded."""
    from tools.tool_registry import resolve_tools

    tools = await resolve_tools(RESEARCH_AGENT_TOOL_IDS)
    return {**RESEARCH_AGENT, "tools": tools}
