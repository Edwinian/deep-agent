"""Rewrite user queries for improved RAG retrieval."""

from __future__ import annotations

from constants.model_name import ModelName

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial query:"
    "\n ------- \n"
    "{query}"
    "\n ------- \n"
    "Formulate an improved query:"
)

_DEFAULT_MODEL = ModelName.GROK_3_MINI.with_provider()


def _get_rewrite_model():
    """Resolve and return the chat model used for query rewriting."""
    from utils.resolve_model import resolve_model

    model = resolve_model(model=_DEFAULT_MODEL)
    if model is None:
        raise RuntimeError("No model configured for rewrite_query")
    return model


def rewrite_query(query: str) -> str:
    """Rewrite a user query to improve vector retrieval.

    Args:
        query: The user's original query.

    Returns:
        An improved query with clearer semantic intent.
    """
    model = _get_rewrite_model()
    prompt = REWRITE_PROMPT.format(query=query)
    response = model.invoke([{"role": "user", "content": prompt}])
    rewritten = str(response.content or "").strip()
    if not rewritten:
        raise RuntimeError("rewrite_query returned empty content")
    return rewritten
