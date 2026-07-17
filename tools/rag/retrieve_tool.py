"""Retrieve indexed documents from ChromaDB."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from langchain.tools import ToolRuntime
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from typing_extensions import Annotated

from chroma_service import ChromaService
from tools.rag.generate_answer import generate_answer
from tools.rag.grade_documents import GradeDocuments, grade_documents
from tools.rag.rewrite_query import rewrite_query
from utils.tool_messages import text_tool_message

RETRIEVE_TOOL_ID = 2008
MAX_QUERY_REWRITES = 1


@lru_cache(maxsize=1)
def _get_retriever() -> BaseRetriever:
    service = ChromaService()
    return service.get_retriever()


def _retrieve_graded_context(
    query: str,
    *,
    original_query: str,
    rewrites_remaining: int,
    on_status: Callable[[str], None] | None = None,
) -> tuple[str | None, str | None]:
    """Retrieve, grade, and optionally rewrite the query before re-retrieving.

    Returns:
        A tuple of (context, error_message). Exactly one value is set.
    """
    if on_status is not None:
        on_status("Retrieving context from vector store…")

    retriever = _get_retriever()
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        return None, "No relevant documents found."

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    if on_status is not None:
        on_status("Grading retrieved context…")

    grade = grade_documents(original_query, context)

    if grade == GradeDocuments.REWRITE_QUERY:
        if rewrites_remaining > 0:
            if on_status is not None:
                on_status("Context irrelevant, retry retrieving context…")
            rewritten_query = rewrite_query(query)
            return _retrieve_graded_context(
                rewritten_query,
                original_query=original_query,
                rewrites_remaining=rewrites_remaining - 1,
                on_status=on_status,
            )
        return (
            None,
            "Retrieved documents were not relevant to the query after rewriting it.",
        )

    return context, None


@tool(parse_docstring=True)
def retrieve_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    runtime: ToolRuntime,
) -> Command:
    """Search and return relevant passages from indexed documents.

    Retrieves from ChromaDB, grades relevance against the query, rewrites and
    retries retrieval when needed, then generates a concise grounded answer.

    Args:
        query: Natural language search query.
        tool_call_id: Injected tool call identifier for message tracking.
        runtime: Injected tool runtime for streaming progress to the client.

    Returns:
        Command that adds the generated answer as a text tool message.
    """

    def _status(message: str) -> None:
        runtime.emit_output_delta(message)

    context, error = _retrieve_graded_context(
        query,
        original_query=query,
        rewrites_remaining=MAX_QUERY_REWRITES,
        on_status=_status,
    )

    if error is not None:
        return Command(
            update={
                "messages": [
                    text_tool_message(error, tool_call_id, status="error"),
                ],
            }
        )

    _status("Retrieved context, now answering user…")
    answer = generate_answer(query, context or "")
    return Command(
        update={
            "messages": [
                text_tool_message(answer, tool_call_id),
            ],
        }
    )
