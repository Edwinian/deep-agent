"""Grade retrieved document relevance for agentic RAG workflows."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field

from constants.function_name import FunctionName
from constants.model_name import ModelName
from utils.tracing import trace

_DEFAULT_MODEL = ModelName.GROK_3_MINI.with_provider()


class GradeDocuments(StrEnum):
    """Next RAG step based on retrieved document relevance."""

    GENERATE_ANSWER = FunctionName.GENERATE_ANSWER
    REWRITE_QUERY = FunctionName.REWRITE_QUERY


class _GradeDocumentsResult(BaseModel):
    """Structured grader output mapped to :class:`GradeDocuments`."""

    route: GradeDocuments = Field(
        description=(
            f"Use {FunctionName.GENERATE_ANSWER} when the retrieved document is relevant to the question; "
            f"otherwise use {FunctionName.REWRITE_QUERY}."
        )
    )


@lru_cache(maxsize=1)
def _get_grade_prompt_template() -> str:
    """Load the grade_documents prompt template from the database."""
    from db.agent_store import get_system_prompt_by_name

    return get_system_prompt_by_name(FunctionName.GRADE_DOCUMENTS).content


def _get_grader_model():
    """Resolve and return the chat model used for document grading."""
    from utils.resolve_model import resolve_model

    model = resolve_model(model=_DEFAULT_MODEL)
    if model is None:
        raise RuntimeError("No model configured for grade_documents")
    return model


@trace(FunctionName.GRADE_DOCUMENTS)
def grade_documents(question: str, context: str) -> GradeDocuments:
    """Assess whether retrieved document chunks are relevant to the user question.

    Args:
        question: The user's original question.
        context: Retrieved document text to grade against the question.

    Returns:
        :class:`GradeDocuments.GENERATE_ANSWER` if the context is relevant,
        otherwise :class:`GradeDocuments.REWRITE_QUERY`.
    """
    grader_model = _get_grader_model()
    prompt = _get_grade_prompt_template().format(question=question, context=context)
    response = grader_model.with_structured_output(_GradeDocumentsResult).invoke(
        [{"role": "user", "content": prompt}]
    )
    return response.route
