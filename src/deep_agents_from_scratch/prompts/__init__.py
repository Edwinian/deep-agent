"""Prompt templates and tool descriptions for deep agents."""

from deep_agents_from_scratch.prompts.file_usage_instructions import (
    FILE_USAGE_INSTRUCTIONS,
)
from deep_agents_from_scratch.prompts.ls_description import LS_DESCRIPTION
from deep_agents_from_scratch.prompts.read_file_description import READ_FILE_DESCRIPTION
from deep_agents_from_scratch.prompts.researcher_instructions import (
    RESEARCHER_INSTRUCTIONS,
)
from deep_agents_from_scratch.prompts.subagent_usage_instructions import (
    SUBAGENT_USAGE_INSTRUCTIONS,
)
from deep_agents_from_scratch.prompts.summarize_web_search import SUMMARIZE_WEB_SEARCH
from deep_agents_from_scratch.prompts.task_description_prefix import (
    TASK_DESCRIPTION_PREFIX,
)
from deep_agents_from_scratch.prompts.todo_usage_instructions import (
    TODO_USAGE_INSTRUCTIONS,
)
from deep_agents_from_scratch.prompts.write_file_description import (
    WRITE_FILE_DESCRIPTION,
)
from deep_agents_from_scratch.prompts.write_todos_description import (
    WRITE_TODOS_DESCRIPTION,
)

__all__ = [
    "FILE_USAGE_INSTRUCTIONS",
    "LS_DESCRIPTION",
    "READ_FILE_DESCRIPTION",
    "RESEARCHER_INSTRUCTIONS",
    "SUBAGENT_USAGE_INSTRUCTIONS",
    "SUMMARIZE_WEB_SEARCH",
    "TASK_DESCRIPTION_PREFIX",
    "TODO_USAGE_INSTRUCTIONS",
    "WRITE_FILE_DESCRIPTION",
    "WRITE_TODOS_DESCRIPTION",
]
