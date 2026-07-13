"""Prompt templates and tool descriptions for deep agents."""

from prompts.file_usage_instructions import (
    FILE_USAGE_INSTRUCTIONS,
)
from prompts.ls_description import LS_DESCRIPTION
from prompts.read_file_description import READ_FILE_DESCRIPTION
from prompts.researcher_instructions import (
    RESEARCHER_INSTRUCTIONS,
)
from prompts.rag_agent_instructions import RAG_AGENT_INSTRUCTIONS
from prompts.subagent_usage_instructions import (
    SUBAGENT_USAGE_INSTRUCTIONS,
)
from prompts.summarize import SUMMARIZE
from prompts.task_description_prefix import (
    TASK_DESCRIPTION_PREFIX,
)
from prompts.todo_usage_instructions import (
    TODO_USAGE_INSTRUCTIONS,
)
from prompts.write_file_description import (
    WRITE_FILE_DESCRIPTION,
)
from prompts.write_todos_description import (
    WRITE_TODOS_DESCRIPTION,
)

__all__ = [
    "FILE_USAGE_INSTRUCTIONS",
    "LS_DESCRIPTION",
    "READ_FILE_DESCRIPTION",
    "RESEARCHER_INSTRUCTIONS",
    "RAG_AGENT_INSTRUCTIONS",
    "SUBAGENT_USAGE_INSTRUCTIONS",
    "SUMMARIZE",
    "TASK_DESCRIPTION_PREFIX",
    "TODO_USAGE_INSTRUCTIONS",
    "WRITE_FILE_DESCRIPTION",
    "WRITE_TODOS_DESCRIPTION",
]
