"""Registry of all feature-module routers mounted by the FastAPI app.

Importing :data:`routers` instantiates one of every controller in declaration
order, so :mod:`main` only needs ``[app.include_router(r) for r in routers]``.
To add a new module, append its controller here — nothing else in ``main``
needs to change.
"""

from __future__ import annotations

from fastapi import APIRouter

from modules.agents import AgentsController
from modules.chats import ChatsController
from modules.skills import SkillsController
from modules.system_prompts import SystemPromptsController

routers: list[APIRouter] = [
    ChatsController().router,
    AgentsController().router,
    SystemPromptsController().router,
    SkillsController().router,
]

__all__ = ["routers"]
