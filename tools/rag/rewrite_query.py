"""Rewrite user queries for improved RAG retrieval."""

from __future__ import annotations

from functools import lru_cache

from constants.function_name import FunctionName
from constants.model_name import ModelName
from utils.tracing import trace

_DEFAULT_MODEL = ModelName.GROK_3_MINI.with_provider()


@lru_cache(maxsize=1)
def _get_rewrite_prompt_template() -> str:
    """Load the rewrite_query prompt template from the database."""
    from db.agent_store import get_system_prompt_by_name

    return get_system_prompt_by_name(FunctionName.REWRITE_QUERY).content


def _get_rewrite_model():
    """Resolve and return the chat model used for query rewriting."""
    from utils.resolve_model import resolve_model

    model = resolve_model(model=_DEFAULT_MODEL)
    if model is None:
        raise RuntimeError("No model configured for rewrite_query")
    return model


@trace(FunctionName.REWRITE_QUERY)
def rewrite_query(query: str) -> str:
    """Rewrite a user query to improve vector retrieval.

    Args:
        query: The user's original query.

    Returns:
        An improved query with clearer semantic intent.
    """
    model = _get_rewrite_model()
    prompt = _get_rewrite_prompt_template().format(query=query)
    response = model.invoke([{"role": "user", "content": prompt}])
    rewritten = str(response.content or "").strip()
    if not rewritten:
        raise RuntimeError("rewrite_query returned empty content")
    return rewritten
