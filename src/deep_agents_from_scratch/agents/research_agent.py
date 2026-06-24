"""Research agent spec."""

from deep_agents_from_scratch.agents.types import DeepAgent
from deep_agents_from_scratch.prompts import RESEARCHER_INSTRUCTIONS
from deep_agents_from_scratch.tools import get_today_str, think_tool, web_search_tool

RESEARCH_AGENT: DeepAgent = {
    "name": "research-agent",
    "description": (
        "Delegate research to the sub-agent researcher. "
        "Only give this researcher one topic at a time."
    ),
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    "tools": [web_search_tool, think_tool],
}
