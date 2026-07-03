"""Abstract LLM service interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LlmService(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Run a text prompt and return the model response."""
