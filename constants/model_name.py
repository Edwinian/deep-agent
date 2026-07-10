"""LLM model name constants."""

from enum import StrEnum


class ModelProvider(StrEnum):
    """LangChain model provider prefixes."""

    XAI = "xai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelName(StrEnum):
    """Supported model identifiers."""

    GROK_3_MINI = "grok-3-mini"
    GROK_4_3 = "grok-4.3"
    GROK_4_FAST_NON_REASONING = "grok-4-fast-non-reasoning"
    ALL_MINI_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"

    def with_provider(self, provider: ModelProvider = ModelProvider.XAI) -> str:
        """Return the LangChain provider-prefixed model string."""
        return f"{provider.value}:{self.value}"
