"""CRUD service for the ``Agent`` table.

Extends :class:`modules.base_service.BaseService` with create / update /
delete on top of the existing read helpers in :mod:`db.agent_store`.
The store opens one short-lived SQLite connection per call; each operation
runs in a worker thread via :func:`asyncio.to_thread` so the event loop is
never blocked.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from db.agent_store import (
    AGENT_DB_CONN_STRING,
    AgentNotFoundError,
    AgentRow,
    _base_row_fields,
    _connect,
    _row_to_agent,
)
from modules.base_service import BaseService
from pydantic import BaseModel, Field


class AgentCreate(BaseModel):
    """Payload for ``POST /agents``."""

    name: str
    description: str
    system_prompt_id: int
    subagent_ids: list[int] | None = None
    model: str | None = None
    tool_ids: list[int] | None = None
    skill_ids: list[int] | None = None


class AgentUpdate(BaseModel):
    """Partial payload for ``PUT /agents/{id}`` — only set fields are applied."""

    name: str | None = None
    description: str | None = None
    system_prompt_id: int | None = None
    subagent_ids: list[int] | None = None
    model: str | None = None
    tool_ids: list[int] | None = None
    skill_ids: list[int] | None = None


def _dump_optional_id_list(value: list[int] | None) -> str | None:
    return json.dumps(value) if value is not None else None


class AgentsService(BaseService[AgentRow, AgentCreate, AgentUpdate]):
    """CRUD operations on the ``Agent`` table."""

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def get_many(self) -> list[AgentRow]:
        return await asyncio.to_thread(self._get_many_sync)

    def _get_many_sync(self) -> list[AgentRow]:
        conn = _connect(AGENT_DB_CONN_STRING)
        try:
            rows = conn.execute(
                """
                SELECT
                    id, name, description, system_prompt_id,
                    subagent_ids, model, tool_ids, skill_ids,
                    created_at, updated_at, deleted_at
                FROM Agent
                WHERE deleted_at IS NULL
                ORDER BY id
                """
            ).fetchall()
            return [_row_to_agent(row) for row in rows]
        finally:
            conn.close()

    async def get_one(self, row_id: int) -> AgentRow:
        return await asyncio.to_thread(self._get_one_sync, row_id)

    def _get_one_sync(self, row_id: int) -> AgentRow:
        conn = _connect(AGENT_DB_CONN_STRING)
        try:
            row = conn.execute(
                """
                SELECT
                    id, name, description, system_prompt_id,
                    subagent_ids, model, tool_ids, skill_ids,
                    created_at, updated_at, deleted_at
                FROM Agent
                WHERE id = ? AND deleted_at IS NULL
                """,
                (row_id,),
            ).fetchone()
            if row is None:
                raise AgentNotFoundError(f"Unknown agent_id: {row_id}")
            return _row_to_agent(row)
        finally:
            conn.close()

    async def create(self, payload: AgentCreate) -> AgentRow:
        return await asyncio.to_thread(self._create_sync, payload)

    def _create_sync(self, payload: AgentCreate) -> AgentRow:
        now = self._now()
        conn = _connect(AGENT_DB_CONN_STRING)
        try:
            cursor = conn.execute(
                """
                INSERT INTO Agent (
                    name, description, system_prompt_id, subagent_ids,
                    model, tool_ids, skill_ids, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.name,
                    payload.description,
                    payload.system_prompt_id,
                    _dump_optional_id_list(payload.subagent_ids),
                    payload.model,
                    _dump_optional_id_list(payload.tool_ids),
                    _dump_optional_id_list(payload.skill_ids),
                    now,
                ),
            )
            conn.commit()
            row_id = int(cursor.lastrowid)
        finally:
            conn.close()
        return self._get_one_sync(row_id)

    async def update(self, row_id: int, payload: AgentUpdate) -> AgentRow:
        return await asyncio.to_thread(self._update_sync, row_id, payload)

    def _update_sync(self, row_id: int, payload: AgentUpdate) -> AgentRow:
        # Read first so a missing row raises before we try to write.
        self._get_one_sync(row_id)

        assignments: list[str] = []
        values: list[object] = []

        mapping: dict[str, object] = {
            "name": payload.name,
            "description": payload.description,
            "system_prompt_id": payload.system_prompt_id,
            "model": payload.model,
        }
        for column, value in mapping.items():
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)

        for column, value in (
            ("subagent_ids", payload.subagent_ids),
            ("tool_ids", payload.tool_ids),
            ("skill_ids", payload.skill_ids),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(_dump_optional_id_list(value))

        if not assignments:
            return self._get_one_sync(row_id)

        assignments.append("updated_at = ?")
        values.append(self._now())
        values.append(row_id)

        conn = _connect(AGENT_DB_CONN_STRING)
        try:
            conn.execute(
                f"UPDATE Agent SET {', '.join(assignments)} WHERE id = ?",
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
                "UPDATE Agent SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
                (now, row_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                raise AgentNotFoundError(f"Unknown agent_id: {row_id}")
        finally:
            conn.close()
