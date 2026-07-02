"""Research agent spec."""

from agents.types import DeepAgent
from prompts import RESEARCHER_INSTRUCTIONS
from tools import get_today_str, think_tool, web_search_tool

RESEARCH_AGENT_ID = 1001

RESEARCH_AGENT: DeepAgent = {
    "id": RESEARCH_AGENT_ID,
    "name": "research-agent",
    "description": (
        "Delegate research to the sub-agent researcher. "
        "Only give this researcher one topic at a time."
    ),
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    "tools": [web_search_tool, think_tool],
}
