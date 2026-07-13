"""Prompt for rewriting RAG retrieval queries."""

from constants.function_name import FunctionName

PROMPT_NAME = FunctionName.REWRITE_QUERY

REWRITE_QUERY = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial query:"
    "\n ------- \n"
    "{query}"
    "\n ------- \n"
    "Formulate an improved query:"
)
