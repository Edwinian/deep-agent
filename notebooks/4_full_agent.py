"""Deep Agent for Research.

Converted from notebooks/4_full_agent.ipynb.
Demonstrates a full research agent combining TODOs, files, sub-agents, and Tavily search.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, cast

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from deep_agents_from_scratch.file_tools import ls, read_file, write_file
from deep_agents_from_scratch.prompts import (
    FILE_USAGE_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)
from deep_agents_from_scratch.research_tools import get_today_str, tavily_search, think_tool
from deep_agents_from_scratch.state import DeepAgentState
from deep_agents_from_scratch.task_tool import SubAgent as ScratchSubAgent, _create_task_tool
from deep_agents_from_scratch.todo_tools import read_todos, write_todos
from deepagents.middleware.subagents import (
    CompiledSubAgent,
    SubAgent as DeepAgentSubAgent,
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


def _create_scratch_agent(model):
    sub_agent_tools = [tavily_search, think_tool]
    built_in_tools = [ls, read_file, write_file, write_todos, read_todos, think_tool]

    research_sub_agent: ScratchSubAgent = {
        "name": "research-agent",
        "description": (
            "Delegate research to the sub-agent researcher. "
            "Only give this researcher one topic at a time."
        ),
        "prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
        "tools": ["tavily_search", "think_tool"],
    }

    task_tool = _create_task_tool(
        sub_agent_tools, [research_sub_agent], model, DeepAgentState
    )
    all_tools = sub_agent_tools + built_in_tools + [task_tool]

    instructions = _build_instructions(_build_subagent_instructions())
    return create_agent(
        model,
        all_tools,
        system_prompt=instructions,
        state_schema=DeepAgentState,
    )


def _run_scratch_agent_demo(model) -> None:
    print("\n=== Scratch deep agent (create_agent) ===\n")

    show_prompt(RESEARCHER_INSTRUCTIONS, title="Researcher Prompt")

    instructions = _build_instructions(_build_subagent_instructions())
    show_prompt(instructions, title="System Prompt")

    agent = _create_scratch_agent(model)

    graph_path = _save_agent_graph(agent)
    print(f"\nAgent graph saved to {graph_path}")

    result = agent.invoke(
        cast(
            Any,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Give me an overview of Model Context Protocol (MCP).",
                    }
                ],
            },
        )
    )

    print("\n--- agent result ---")
    format_messages(result["messages"])


def _run_deepagents_demo(model) -> None:
    from deepagents import create_deep_agent

    print("\n=== Deep Agent package (create_deep_agent) ===\n")

    sub_agent_tools = [tavily_search, think_tool]
    instructions = _build_instructions(_build_subagent_instructions())

    research_sub_agent = cast(
        DeepAgentSubAgent,
        {
            "name": "research-agent",
            "description": (
                "Delegate research to the sub-agent researcher. "
                "Only give this researcher one topic at a time."
            ),
            "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
            "tools": [tavily_search, think_tool],
        },
    )

    agent = create_deep_agent(
        tools=sub_agent_tools,
        system_prompt=instructions,
        subagents=cast(list[DeepAgentSubAgent | CompiledSubAgent], [research_sub_agent]),
        model=model,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Give me an very brief overview of Model Context Protocol (MCP)."
                    ),
                }
            ],
        }
    )

    print("\n--- agent result ---")
    format_messages(result["messages"])


def main() -> None:
    model = init_chat_model(model=MODEL, temperature=0.0)

    _run_scratch_agent_demo(model)
    _run_deepagents_demo(model)


if __name__ == "__main__":
    main()
