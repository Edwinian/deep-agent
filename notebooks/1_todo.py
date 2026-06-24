"""Planning: TODO Lists.

Converted from notebooks/1_todo.ipynb.
Demonstrates TODO list tools for task planning and progress tracking in agents.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from deep_agents_from_scratch.prompts import (
    TODO_USAGE_INSTRUCTIONS,
    WRITE_TODOS_DESCRIPTION,
)
from deep_agents_from_scratch.state import DeepAgentState
from scratch.todo_tools import read_todos, write_todos
from utils import format_messages, show_prompt

MODEL = "xai:grok-3-mini"

SEARCH_RESULT = """The Model Context Protocol (MCP) is an open standard protocol developed
by Anthropic to enable seamless integration between AI models and external systems like
tools, databases, and other services. It acts as a standardized communication layer,
allowing AI models to access and utilize data from various sources in a consistent and
efficient manner. Essentially, MCP simplifies the process of connecting AI assistants
to external services by providing a unified language for data exchange. """

SIMPLE_RESEARCH_INSTRUCTIONS = """IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the user's question."""

warnings.filterwarnings(
    "ignore",
    message="LangSmith now uses UUID v7",
    category=UserWarning,
)

load_dotenv(PROJECT_ROOT / ".env", override=True)


@tool(parse_docstring=True)
def web_search(query: str) -> str:
    """Search the web for information on a specific topic.

    This tool performs web searches and returns relevant results
    for the given query. Use this when you need to gather information from
    the internet about any topic.

    Args:
        query: The search query string. Be specific and clear about what
               information you're looking for.

    Returns:
        Search results from search engine.

    Example:
        web_search("machine learning applications in healthcare")
    """
    del query
    return SEARCH_RESULT


def _save_agent_graph(agent, filename: str = "1_todo_graph.png") -> Path:
    graph_path = NOTEBOOK_DIR / "assets" / filename
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_bytes(agent.get_graph(xray=True).draw_mermaid_png())
    return graph_path


def main() -> None:
    print("--- write_todos tool description ---")
    show_prompt(WRITE_TODOS_DESCRIPTION, title="write_todos")

    print("\n--- todo usage instructions ---")
    show_prompt(TODO_USAGE_INSTRUCTIONS, title="System Prompt")

    model = init_chat_model(model=MODEL, temperature=0.0)
    tools = [write_todos, web_search, read_todos]

    agent = create_agent(
        model,
        tools,
        system_prompt=TODO_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + SIMPLE_RESEARCH_INSTRUCTIONS,
        state_schema=DeepAgentState,
    )

    graph_path = _save_agent_graph(agent)
    print(f"\nAgent graph saved to {graph_path}")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Give me a short summary of the Model Context Protocol (MCP).",
                }
            ],
            "todos": [],
        }
    )

    print("\n--- agent result ---")
    format_messages(result["messages"])


if __name__ == "__main__":
    main()
