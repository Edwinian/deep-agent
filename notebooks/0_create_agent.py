"""Create Agent - Prebuilt.

Converted from notebooks/0_create_agent.ipynb.
Demonstrates building a ReAct agent with create_agent, tools, and custom state.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Annotated, List, Literal, Union

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, messages_to_dict
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from utils import format_messages

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent

warnings.filterwarnings(
    "ignore",
    message="LangSmith now uses UUID v7",
    category=UserWarning,
)

load_dotenv(PROJECT_ROOT / ".env", override=True)


@tool
def calculator(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: Union[int, float],
    b: Union[int, float],
) -> Union[int, float]:
    """Define a two-input calculator tool that returns precise answers.

    Arg:
        operation (str): The operation to perform ('add', 'subtract', 'multiply', 'divide').
        a (float or int): The first number.
        b (float or int): The second number.

    Returns:
        result (float or int): the result of the operation
    Example
        Divide: result   = a / b
        Subtract: result = a - b
    """
    if operation == "divide" and b == 0:
        return {"error": "Division by zero is not allowed."}

    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b
    else:
        result = "unknown operation"
    return result


def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling cases where either or both inputs might be None."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right


class CalcState(AgentState):
    """Graph State."""

    ops: Annotated[List[str], reduce_list]


@tool
def calculator_wstate(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: Union[int, float],
    b: Union[int, float],
    state: Annotated[CalcState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Union[int, float]:
    """Define a two-input calculator tool.

    Arg:
        operation (str): The operation to perform ('add', 'subtract', 'multiply', 'divide').
        a (float or int): The first number.
        b (float or int): The second number.

    Returns:
        result (float or int): the result of the operation
    Example
        Divide: result   = a / b
        Subtract: result = a - b
    """
    del state  # injected by LangGraph; not used in this example
    if operation == "divide" and b == 0:
        return {"error": "Division by zero is not allowed."}

    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b
    else:
        result = "unknown operation"
    ops = [f"({operation}, {a}, {b}),"]
    return Command(
        update={
            "ops": ops,
            "messages": [
                ToolMessage(f"{result}", tool_call_id=tool_call_id)
            ],
        }
    )


SYSTEM_PROMPT = """
You are a helpful arithmetic assistant who is an expert at using a calculator.
Return all text as plain text without Markdown math delimiters.
"""

MODEL = "xai:grok-3-mini"


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def _save_agent_graph(agent, filename: str = "0_create_agent_graph.png") -> Path:
    graph_path = NOTEBOOK_DIR / "assets" / filename
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_bytes(agent.get_graph(xray=True).draw_mermaid_png())
    return graph_path


def main() -> None:
    # --- Basic calculator agent ---
    model = init_chat_model(model=MODEL, temperature=0.0)
    agent = create_agent(
        model,
        [calculator],
        system_prompt=SYSTEM_PROMPT,
    ).with_config({"recursion_limit": 20})

    graph_path = _save_agent_graph(agent)
    print(f"Agent graph saved to {graph_path}")
    print(f"Agent type: {type(agent)}")

    result1 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 3.1 * 4.2?",
                }
            ],
        }
    )
    print("\n--- result1 messages ---")
    format_messages(result1["messages"])
    print("\n--- result1 as JSON ---")
    _print_json({"messages": messages_to_dict(result1["messages"])})

    # --- Calculator with custom state ---
    model = init_chat_model(model=MODEL, temperature=0.0)
    agent = create_agent(
        model,
        [calculator_wstate],
        system_prompt=SYSTEM_PROMPT,
        state_schema=CalcState,
    ).with_config({"recursion_limit": 20})

    result2 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 3.1 * 4.2?",
                }
            ],
        }
    )
    print("\n--- result2 messages ---")
    format_messages(result2["messages"])
    print("\n--- result2 (includes ops) ---")
    _print_json(result2)

    result3 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 3.1 * 4.2 + 5.5 * 6.5?",
                }
            ],
        }
    )
    print("\n--- result3 messages ---")
    format_messages(result3["messages"])
    print("\n--- result3 (includes ops) ---")
    _print_json(result3)


if __name__ == "__main__":
    main()
