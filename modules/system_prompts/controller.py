"""REST controller exposing ``SystemPromptsService`` CRUD at ``/system-prompts``."""

from __future__ import annotations

from db.agent_store import AgentNotFoundError
from modules.base_controller import BaseController
from modules.controller_utils import row_not_found
from modules.system_prompts.service import (
    SystemPromptCreate,
    SystemPromptUpdate,
    SystemPromptsService,
)


class SystemPromptsController(BaseController):
    """Mounts ``/system-prompts`` routes that delegate to :class:`SystemPromptsService`."""

    PREFIX = "/system-prompts"

    def __init__(self) -> None:
        self.service = SystemPromptsService()
        super().__init__()

    def _register_routes(self) -> None:
        router = self.router
        service = self.service

        @router.get("")
        async def list_system_prompts() -> list[dict]:
            rows = await service.get_many()
            return [row.__dict__ for row in rows]

        @router.get("/{prompt_id}")
        async def get_system_prompt(prompt_id: int) -> dict:
            try:
                row = await service.get_one(prompt_id)
            except AgentNotFoundError as exc:
                raise row_not_found(exc) from exc
            return row.__dict__

        @router.post("", status_code=201)
        async def create_system_prompt(payload: SystemPromptCreate) -> dict:
            row = await service.create(payload)
            return row.__dict__

        @router.put("/{prompt_id}")
        async def update_system_prompt(
            prompt_id: int, payload: SystemPromptUpdate
        ) -> dict:
            try:
                row = await service.update(prompt_id, payload)
            except AgentNotFoundError as exc:
                raise row_not_found(exc) from exc
            return row.__dict__

        @router.delete("/{prompt_id}", status_code=204)
        async def delete_system_prompt(prompt_id: int) -> None:
            try:
                await service.delete(prompt_id)
            except AgentNotFoundError as exc:
                raise row_not_found(exc) from exc
