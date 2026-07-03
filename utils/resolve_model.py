"""Resolve DeepAgent model settings into a chat model."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from agents.types import DeepAgent, ModelConfig, model_config_kwargs


def resolve_model(
    agent: DeepAgent | None = None,
    *,
    model: str | BaseChatModel | None = None,
    model_config: ModelConfig | None = None,
) -> str | BaseChatModel | None:
    """Resolve the model for an agent spec or compile-time overrides."""
    resolved_config = (
        model_config
        if model_config is not None
        else agent.get("model_config") if agent is not None else None
    )
    if resolved_config is not None:
        return init_chat_model(**model_config_kwargs(resolved_config))
    if agent is None:
        if model is None or isinstance(model, BaseChatModel):
            return model
        return init_chat_model(model=model)
    return agent.get("model", model)
