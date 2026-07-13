"""Generate concise answers from retrieved RAG context."""

from __future__ import annotations

from functools import lru_cache

from constants.function_name import FunctionName
from constants.model_name import ModelName
from utils.tracing import trace

_DEFAULT_MODEL = ModelName.GROK_3_MINI.with_provider()


@lru_cache(maxsize=1)
def _get_generate_prompt_template() -> str:
    """Load the generate_answer prompt template from the database."""
    from db.agent_store import get_system_prompt_by_name

    return get_system_prompt_by_name(FunctionName.GENERATE_ANSWER).content


def _get_answer_model():
    """Resolve and return the chat model used for answer generation."""
    from utils.resolve_model import resolve_model

    model = resolve_model(model=_DEFAULT_MODEL)
    if model is None:
        raise RuntimeError("No model configured for generate_answer")
    return model


@trace(FunctionName.GENERATE_ANSWER)
def generate_answer(query: str, context: str) -> str:
    """Generate an answer from a query and retrieved context.

    Args:
        query: The user's question.
        context: Retrieved document text to answer from.

    Returns:
        A concise answer grounded in the retrieved context.
    """
    model = _get_answer_model()
    prompt = _get_generate_prompt_template().format(query=query, context=context)
    response = model.invoke([{"role": "user", "content": prompt}])
    answer = str(response.content or "").strip()
    if not answer:
        raise RuntimeError("generate_answer returned empty content")
    return answer
