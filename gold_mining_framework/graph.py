"""Linear LangGraph pipeline of specialist agents.

The outer graph has one-way edges only. Each node is an agent. Additional
agents can be appended after idea_generation_agent later.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from gold_mining_framework.agents import idea_generation_agent
from gold_mining_framework.state import AppIdeaState


def build_graph() -> CompiledStateGraph:
    """Compile the linear gold mining framework graph.

    Current topology::

        START -> idea_generation_agent -> END

    Returns:
        Compiled outer graph. Invoke with ``{"query": "<market>"}``.
    """
    builder = StateGraph(AppIdeaState)
    builder.add_node("idea_generation_agent", idea_generation_agent)
    builder.add_edge(START, "idea_generation_agent")
    builder.add_edge("idea_generation_agent", END)
    return builder.compile()


# Compiled pipeline: START -> idea_generation_agent -> END
graph = build_graph()
