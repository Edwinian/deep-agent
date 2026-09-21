"""ReAct harness for the idea generation agent (LLM ⇄ web_search)."""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from gold_mining_framework.agents.idea_generation_agent.prompt import PROMPT
from gold_mining_framework.llm import get_chat_model
from gold_mining_framework.tools import web_search

logger = logging.getLogger(__name__)

# Inner ReAct loops may issue many trend-check searches.
RECURSION_LIMIT = 80


def _message_text(message: AIMessage) -> str:
    """Extract plain text from an AI message body.

    Args:
        message: Model response that may store content as a string or blocks.

    Returns:
        Concatenated text content.
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return str(content)


def build_harness() -> CompiledStateGraph:
    """Build the idea generation ReAct graph.

    The subgraph is ``agent`` ⇄ ``tools`` until the model stops calling
    ``web_search``.

    Returns:
        Compiled LangGraph agent that expects ``messages`` in state.
    """
    tools = [web_search]
    bound_model = None

    def call_model(state: MessagesState) -> dict:
        """Invoke the idea generation LLM with its system prompt."""
        nonlocal bound_model
        if bound_model is None:
            bound_model = get_chat_model().bind_tools(tools)
        response = bound_model.invoke(
            [SystemMessage(content=PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


def _log_progress(update: dict) -> None:
    """Log tool calls and search completions so long runs are not silent."""
    if "agent" in update:
        messages = update["agent"].get("messages") or []
        if not messages:
            return
        message = messages[-1]
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            for tool_call in tool_calls:
                query = (tool_call.get("args") or {}).get("query", "")
                logger.info("web_search: %s", query or tool_call.get("name"))
        else:
            logger.info("Writing market hierarchy...")
    if "tools" in update:
        logger.info("Received search results.")


def run_harness(query: str, harness: CompiledStateGraph) -> tuple[list, str]:
    """Run the idea generation harness on a market query.

    Args:
        query: Market, category, or focus area from ``graph.invoke``.
        harness: Compiled idea generation subgraph.

    Returns:
        A tuple of (message trace, final market hierarchy text).
    """
    messages: list = [HumanMessage(content=query)]
    for update in harness.stream(
        {"messages": [HumanMessage(content=query)]},
        {"recursion_limit": RECURSION_LIMIT},
        stream_mode="updates",
    ):
        _log_progress(update)
        for node_update in update.values():
            messages.extend(node_update.get("messages") or [])

    if len(messages) < 2:
        raise RuntimeError("Idea generation agent produced no output.")

    final = messages[-1]
    hierarchy = (
        _message_text(final) if isinstance(final, AIMessage) else str(final.content)
    )
    return messages, hierarchy
