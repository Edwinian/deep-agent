"""Deep Agent for Research.

Converted from notebooks/4_full_agent.ipynb.
Demonstrates a full research agent using create_deep_agent with Tavily search.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import cast

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from deepagents.middleware.subagents import (
    CompiledSubAgent,
    SubAgent as DeepAgentSubAgent,
)

from deep_agents_from_scratch.agents import RESEARCH_AGENT
from deep_agents_from_scratch.prompts import (
    FILE_USAGE_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)
from utils import format_messages, show_prompt

MODEL = "xai:grok-3-mini"
MAX_CONCURRENT_RESEARCH_UNITS = 3
MAX_RESEARCHER_ITERATIONS = 3

warnings.filterwarnings(
    "ignore",
    message="LangSmith now uses UUID v7",
    category=UserWarning,
)

load_dotenv(PROJECT_ROOT / ".env", override=True)


def _save_agent_graph(agent, filename: str = "4_full_agent_graph.png") -> Path:
    graph_path = NOTEBOOK_DIR / "assets" / filename
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_bytes(agent.get_graph(xray=True).draw_mermaid_png())
    return graph_path


def _build_subagent_instructions() -> str:
    return SUBAGENT_USAGE_INSTRUCTIONS.format(
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
        max_researcher_iterations=MAX_RESEARCHER_ITERATIONS,
        date=datetime.now().strftime("%a %b %-d, %Y"),
    )


def _build_instructions(subagent_instructions: str) -> str:
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


def main() -> None:
    show_prompt(RESEARCHER_INSTRUCTIONS, title="Researcher Prompt")

    instructions = _build_instructions(_build_subagent_instructions())
    show_prompt(instructions, title="System Prompt")

    model = init_chat_model(model=MODEL, temperature=0.0)
    agent = create_deep_agent(
        tools=RESEARCH_AGENT["tools"],
        system_prompt=instructions,
        subagents=cast(
            list[DeepAgentSubAgent | CompiledSubAgent], [RESEARCH_AGENT]
        ),
        model=model,
    )

    graph_path = _save_agent_graph(agent)
    print(f"\nAgent graph saved to {graph_path}")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Give me an overview of Model Context Protocol (MCP).",
                }
            ],
        }
    )

    print("\n--- agent result ---")
    format_messages(result["messages"])


if __name__ == "__main__":
    main()
