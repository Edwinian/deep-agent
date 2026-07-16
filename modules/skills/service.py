"""CRUD service for the ``Skill`` table.

Mirrors :mod:`modules.system_prompts.service` with an extra ``description``
column.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from db.agent_store import (
    AGENT_DB_CONN_STRING,
    SkillNotFoundError,
    SkillRow,
    _base_row_fields,
    _connect,
)
from modules.base_service import BaseService
from pydantic import BaseModel


class SkillCreate(BaseModel):
    name: str
    description: str
    content: str


class SkillUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    content: str | None = None


class SkillsService(BaseService[SkillRow, SkillCreate, SkillUpdate]):
    """CRUD operations on the ``Skill`` table."""

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def get_many(self) -> list[SkillRow]:
        return await asyncio.to_thread(self._get_many_sync)

    def _get_many_sync(self) -> list[SkillRow]:
        conn = _connect(AGENT_DB_CONN_STRING)
        try:
            rows = conn.execute(
                """
                SELECT id, name, description, content, created_at, updated_at, deleted_at
                FROM Skill
                WHERE deleted_at IS NULL
                ORDER BY id
                """
            ).fetchall()
            return [
                SkillRow(
                    **_base_row_fields(row),
                    name=str(row["name"]),
                    description=str(row["description"]),
                    content=str(row["content"]),
                )
                for row in rows
            ]
        finally:
            conn.close()

    async def get_one(self, row_id: int) -> SkillRow:
        return await asyncio.to_thread(self._get_one_sync, row_id)

    def _get_one_sync(self, row_id: int) -> SkillRow:
        conn = _connect(AGENT_DB_CONN_STRING)
        try:
            row = conn.execute(
                """
                SELECT id, name, description, content, created_at, updated_at, deleted_at
                FROM Skill
                WHERE id = ? AND deleted_at IS NULL
                """,
                (row_id,),
            ).fetchone()
            if row is None:
                raise SkillNotFoundError(f"Unknown skill_id: {row_id}")
            return SkillRow(
                **_base_row_fields(row),
                name=str(row["name"]),
                description=str(row["description"]),
                content=str(row["content"]),
            )
        finally:
            conn.close()

    async def create(self, payload: SkillCreate) -> SkillRow:
        return await asyncio.to_thread(self._create_sync, payload)

    def _create_sync(self, payload: SkillCreate) -> SkillRow:
        now = self._now()
        conn = _connect(AGENT_DB_CONN_STRING)
        try:
            cursor = conn.execute(
                """
                INSERT INTO Skill (name, description, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (payload.name, payload.description, payload.content, now),
            )
            conn.commit()
            row_id = int(cursor.lastrowid)
        finally:
            conn.close()
        return self._get_one_sync(row_id)

    async def update(self, row_id: int, payload: SkillUpdate) -> SkillRow:
        return await asyncio.to_thread(self._update_sync, row_id, payload)

    def _update_sync(self, row_id: int, payload: SkillUpdate) -> SkillRow:
        self._get_one_sync(row_id)  # raises if missing

        assignments: list[str] = []
        values: list[object] = []
        for column, value in (
            ("name", payload.name),
            ("description", payload.description),
            ("content", payload.content),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)

        if not assignments:
            return self._get_one_sync(row_id)

        assignments.append("updated_at = ?")
        values.append(self._now())
        values.append(row_id)

        conn = _connect(AGENT_DB_CONN_STRING)
        try:
            conn.execute(
                f"UPDATE Skill SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            conn.commit()
        finally:
            conn.close()
        return self._get_one_sync(row_id)

    async def delete(self, row_id: int) -> None:
        await asyncio.to_thread(self._delete_sync, row_id)

    def _delete_sync(self, row_id: int) -> None:
        now = self._now()
        conn = _connect(AGENT_DB_CONN_STRING)
        try:
            cursor = conn.execute(
                "UPDATE Skill SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
                (now, row_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                raise SkillNotFoundError(f"Unknown skill_id: {row_id}")
        finally:
            conn.close()
