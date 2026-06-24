"""Agent type definitions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NotRequired

from deepagents.middleware.subagents import CompiledSubAgent, SubAgent as DeepAgentSubAgent
from langchain_core.tools import BaseTool

DeepAgentTools = Sequence[BaseTool | Callable | dict[str, Any]]


class DeepAgent(DeepAgentSubAgent, total=False):
    """Agent spec with optional tools and optional nested subagents."""

    tools: NotRequired[DeepAgentTools | None]
    """Optional tools. Defaults to None (middleware-only agents)."""

    subagents: NotRequired[list[DeepAgent | CompiledSubAgent] | None]
    """Optional nested agents for delegation."""
