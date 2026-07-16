"""REST controller exposing ``AgentsService`` CRUD at ``/agents``."""

from __future__ import annotations

from db.agent_store import AgentNotFoundError
from modules.agents.service import (
    AgentCreate,
    AgentUpdate,
    AgentsService,
)
from modules.base_controller import BaseController
from modules.controller_utils import row_not_found


class AgentsController(BaseController):
    """Mounts ``/agents`` routes that delegate to :class:`AgentsService`."""

    PREFIX = "/agents"

    def __init__(self) -> None:
        self.service = AgentsService()
        super().__init__()

    def _register_routes(self) -> None:
        router = self.router
        service = self.service

        @router.get("")
        async def list_agents() -> list[dict]:
            rows = await service.get_many()
            return [row.__dict__ for row in rows]

        @router.get("/{agent_id}")
        async def get_agent(agent_id: int) -> dict:
            try:
                row = await service.get_one(agent_id)
            except AgentNotFoundError as exc:
                raise row_not_found(exc) from exc
            return row.__dict__

        @router.post("", status_code=201)
        async def create_agent(payload: AgentCreate) -> dict:
            row = await service.create(payload)
            return row.__dict__

        @router.put("/{agent_id}")
        async def update_agent(agent_id: int, payload: AgentUpdate) -> dict:
            try:
                row = await service.update(agent_id, payload)
            except AgentNotFoundError as exc:
                raise row_not_found(exc) from exc
            return row.__dict__

        @router.delete("/{agent_id}", status_code=204)
        async def delete_agent(agent_id: int) -> None:
            try:
                await service.delete(agent_id)
            except AgentNotFoundError as exc:
                raise row_not_found(exc) from exc
