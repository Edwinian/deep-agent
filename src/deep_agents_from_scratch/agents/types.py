"""Agent type definitions."""

from __future__ import annotations

from typing import NotRequired, Required

from deepagents.middleware.subagents import CompiledSubAgent, SubAgent

DeepAgentSubAgent = SubAgent


class DeepAgent(SubAgent, total=False):
    """Agent spec with optional nested subagents."""

    id: Required[int]
    subagents: NotRequired[list[DeepAgent | CompiledSubAgent] | None]
    """Optional nested agents for delegation."""
