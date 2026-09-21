"""Gold mining framework pipeline built with plain LangGraph.

The outer graph is linear (one-way edges). Each node is an agent. The first
node is the idea generation agent.
"""

import warnings

import langchain_core  # noqa: F401
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

# LangGraph constructs ``Reviver()`` without ``allowed_objects``. langchain_core
# unmutes that pending deprecation on import, so this filter must run after.
warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change",
    category=LangChainPendingDeprecationWarning,
)

from gold_mining_framework.graph import build_graph, graph  # noqa: E402

__all__ = ["build_graph", "graph"]
