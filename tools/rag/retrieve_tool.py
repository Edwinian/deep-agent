"""Retrieve indexed documents from Qdrant Cloud."""

from __future__ import annotations

from functools import lru_cache

from langchain.tools import ToolRuntime
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from typing_extensions import Annotated

from qdrant_service import QdrantService
from tools.rag.generate_answer import generate_answer
from utils.retry import retry_with_backoff
from utils.tool_messages import text_tool_message
from utils.tool_quality_retry import run_with_quality_retry

RETRIEVE_TOOL_ID = 2008
MAX_QUERY_REWRITES = 1


@lru_cache(maxsize=1)
def _get_retriever() -> BaseRetriever:
    service = QdrantService()
    return service.get_retriever()


def _retrieve_context(query: str) -> str:
    """Fetch and join Qdrant chunks for ``query`` (infra retries only)."""
    retriever = _get_retriever()
    retrieved_docs = retry_with_backoff(lambda: retriever.invoke(query))
    if not retrieved_docs:
        return ""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


@tool(parse_docstring=True)
def retrieve_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    runtime: ToolRuntime,
) -> Command:
    """Search and return relevant passages from indexed documents.

    Retrieves from Qdrant, evaluates output quality against the query, rewrites
    and retries when quality is low, then generates a concise grounded answer.

    Args:
        query: Natural language search query.
        tool_call_id: Injected tool call identifier for message tracking.
        runtime: Injected tool runtime for streaming progress to the client.

    Returns:
        Command that adds the generated answer as a text tool message.
    """

    def _status(message: str) -> None:
        runtime.emit_output_delta(message)

    quality = run_with_quality_retry(
        query,
        _retrieve_context,
        tool_name="retrieve_tool",
        max_retries=MAX_QUERY_REWRITES,
        on_status=_status,
    )

    if not quality.ok:
        return Command(
            update={
                "messages": [
                    text_tool_message(
                        quality.error
                        or "No relevant documents found.",
                        tool_call_id,
                        status="error",
                    ),
                ],
            }
        )

    _status("Retrieved context, now answering user…")
    answer = generate_answer(query, quality.output)
    return Command(
        update={
            "messages": [
                text_tool_message(answer, tool_call_id),
            ],
        }
    )
