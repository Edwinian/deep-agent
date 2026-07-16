"""Feature modules that group related API endpoints into mounted routers."""

from modules.agents import AgentsController
from modules.chats import ChatsController
from modules.skills import SkillsController
from modules.system_prompts import SystemPromptsController

__all__ = [
    "AgentsController",
    "ChatsController",
    "SkillsController",
    "SystemPromptsController",
]
