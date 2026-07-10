"""Summarize text content using an LLM."""

from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from constants.model_name import ModelName
from prompts import SUMMARIZE

_DEFAULT_MODEL = ModelName.GROK_4_3.with_provider()

SUMMARIZE_TOOL_ID = 2005


class Summary(BaseModel):
    """Structured summary with filename and brief text."""

    filename: str = Field(description="Suggested descriptive filename.")
    summary: str = Field(description="Brief summary under 150 words.")


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


def _get_summarize_model():
    """Resolve and return the chat model used for summarization."""
    from utils.resolve_model import resolve_model

    model = resolve_model(model=_DEFAULT_MODEL)
    if model is None:
        raise RuntimeError("No model configured for summarize_tool")
    return model


def summarize_content(content: str) -> Summary:
    """Summarize source content using the configured model and SUMMARIZE prompt."""
    model = _get_summarize_model()
    structured_model = model.with_structured_output(Summary)
    return structured_model.invoke(
        [
            HumanMessage(
                content=SUMMARIZE.format(content=content, date=get_today_str()),
            )
        ],
    )


@tool(parse_docstring=True)
def summarize_tool(content: str) -> Summary:
    """Create a concise summary of source material.

    Capture what the content is about and what an agent should know before
    reading the full text—not every detail. The summary covers the main topic
    or purpose, the type of content (documentation, news, tutorial, reference,
    analysis, etc.), and the most important takeaways in under 150 words.

    Args:
        content: Source text to summarize.

    Returns:
        Structured output with a suggested filename and brief summary.
    """
    return summarize_content(content)
