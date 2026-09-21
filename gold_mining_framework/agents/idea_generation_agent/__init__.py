"""Idea generation agent node for the linear gold mining graph."""

from gold_mining_framework.agents.idea_generation_agent.harness import (
    build_harness,
    run_harness,
)
from gold_mining_framework.state import AppIdeaState

_harness = None


def idea_generation_agent(state: AppIdeaState) -> dict:
    """Run the idea generation agent on the incoming query."""
    global _harness
    if _harness is None:
        _harness = build_harness()

    query = (state.get("query") or "").strip()
    if not query:
        msg = "Invoke the graph with {'query': '<market or focus area>'}."
        raise ValueError(msg)

    messages, hierarchy = run_harness(query, _harness)
    return {"messages": messages, "market_hierarchy": hierarchy}


__all__ = ["idea_generation_agent"]
