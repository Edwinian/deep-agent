"""Agent type definitions."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, Required, TypedDict

from deepagents.middleware.subagents import CompiledSubAgent, SubAgent

DeepAgentSubAgent = SubAgent


class ModelConfig(TypedDict):
    """Keyword arguments for ``langchain.chat_models.init_chat_model``."""

    model: NotRequired[str | None]
    model_provider: NotRequired[str | None]
    configurable_fields: NotRequired[
        Literal["any"] | list[str] | tuple[str, ...] | None
    ]
    config_prefix: NotRequired[str | None]
    temperature: NotRequired[float]
    max_tokens: NotRequired[int]
    timeout: NotRequired[float | int]
    max_retries: NotRequired[int]
    base_url: NotRequired[str]


class DeepAgent(SubAgent, total=False):
    """Agent spec with optional nested subagents.

    Extends :class:`SubAgent` with :class:`DeepAgentField` members.
    """

    id: Required[int]
    """Agent registry id (:attr:`DeepAgentField.ID`)."""
    subagents: NotRequired[list[DeepAgent | CompiledSubAgent] | None]
    """Optional nested agents for delegation (:attr:`DeepAgentField.SUBAGENTS`)."""
    model_config: NotRequired[ModelConfig | None]
    """Optional model settings used when compiling the agent (:attr:`DeepAgentField.MODEL_CONFIG`)."""


def model_config_kwargs(config: ModelConfig) -> dict[str, Any]:
    """Return kwargs for ``init_chat_model``, omitting unset fields."""
    return {key: value for key, value in config.items() if value is not None}
