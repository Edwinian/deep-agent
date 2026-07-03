"""LLM service enums."""

from enum import StrEnum


class LlmErrorPrompt(StrEnum):
    """Prompt fragments and response markers for LLM error handling."""

    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    LENGTH_EXCEEDED = "LENGTH_EXCEEDED"
