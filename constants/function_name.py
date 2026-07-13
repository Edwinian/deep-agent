"""RAG function and system-prompt name constants."""

from enum import StrEnum


class FunctionName(StrEnum):
    """Names for RAG functions and their SystemPrompt rows."""

    GENERATE_ANSWER = "generate_answer"
    GRADE_DOCUMENTS = "grade_documents"
    REWRITE_QUERY = "rewrite_query"
