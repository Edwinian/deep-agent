"""REST controller exposing ``SkillsService`` CRUD at ``/skills``."""

from __future__ import annotations

from db.agent_store import SkillNotFoundError
from modules.base_controller import BaseController
from modules.controller_utils import row_not_found
from modules.skills.service import SkillCreate, SkillUpdate, SkillsService


class SkillsController(BaseController):
    """Mounts ``/skills`` routes that delegate to :class:`SkillsService`."""

    PREFIX = "/skills"

    def __init__(self) -> None:
        self.service = SkillsService()
        super().__init__()

    def _register_routes(self) -> None:
        router = self.router
        service = self.service

        @router.get("")
        async def list_skills() -> list[dict]:
            rows = await service.get_many()
            return [row.__dict__ for row in rows]

        @router.get("/{skill_id}")
        async def get_skill(skill_id: int) -> dict:
            try:
                row = await service.get_one(skill_id)
            except SkillNotFoundError as exc:
                raise row_not_found(exc) from exc
            return row.__dict__

        @router.post("", status_code=201)
        async def create_skill(payload: SkillCreate) -> dict:
            row = await service.create(payload)
            return row.__dict__

        @router.put("/{skill_id}")
        async def update_skill(skill_id: int, payload: SkillUpdate) -> dict:
            try:
                row = await service.update(skill_id, payload)
            except SkillNotFoundError as exc:
                raise row_not_found(exc) from exc
            return row.__dict__

        @router.delete("/{skill_id}", status_code=204)
        async def delete_skill(skill_id: int) -> None:
            try:
                await service.delete(skill_id)
            except SkillNotFoundError as exc:
                raise row_not_found(exc) from exc
