"""SQLite persistence for agent configuration."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DB_CONN_STRING = os.getenv(
    "AGENT_DB_CONN_STRING",
    str(_PROJECT_ROOT / "data" / "agents.db"),
)


class AgentNotFoundError(LookupError):
    """Raised when no agent row exists for the requested ID."""


@dataclass(frozen=True)
class SystemPromptRow:
    id: int
    content: str


@dataclass(frozen=True)
class AgentRow:
    id: int
    name: str
    description: str
    system_prompt_id: int
    subagent_ids: list[int] | None
    model: str | None
    tool_ids: list[int] | None


def _ensure_parent_dir(conn_string: str) -> None:
    db_path = Path(conn_string)
    if db_path.suffix:
        db_path.parent.mkdir(parents=True, exist_ok=True)


def _connect(conn_string: str | None = None) -> sqlite3.Connection:
    resolved = conn_string or AGENT_DB_CONN_STRING
    _ensure_parent_dir(resolved)
    conn = sqlite3.connect(resolved, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_id_list(raw: str, *, field_name: str) -> list[int]:
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must be a JSON array, got: {raw!r}")
    return [int(value) for value in parsed]


def _parse_optional_id_list(
    raw: str | None,
    *,
    field_name: str,
) -> list[int] | None:
    if raw is None:
        return None
    return _parse_id_list(raw, field_name=field_name)


def _row_to_agent(row: sqlite3.Row) -> AgentRow:
    return AgentRow(
        id=int(row["id"]),
        name=str(row["name"]),
        description=str(row["description"]),
        system_prompt_id=int(row["system_prompt_id"]),
        subagent_ids=_parse_optional_id_list(
            row["subagent_ids"],
            field_name="subagent_ids",
        ),
        model=str(row["model"]) if row["model"] is not None else None,
        tool_ids=_parse_optional_id_list(row["tool_ids"], field_name="tool_ids"),
    )


def _agent_table_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(Agent)").fetchall()
    return {str(row["name"]) for row in rows}


def _column_is_not_null(conn: sqlite3.Connection, column_name: str) -> bool:
    rows = conn.execute("PRAGMA table_info(Agent)").fetchall()
    for row in rows:
        if str(row["name"]) == column_name:
            return bool(row["notnull"])
    return False


def _migrate_agent_table(conn: sqlite3.Connection) -> None:
    """Migrate legacy Agent schema to nullable subagent_ids and tool_ids."""
    columns = _agent_table_columns(conn)
    if "subagent_ids" not in columns and "subagent_id" in columns:
        conn.executescript(
            """
            CREATE TABLE Agent_new (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                system_prompt_id INTEGER NOT NULL REFERENCES SystemPrompt(id),
                subagent_ids TEXT,
                model TEXT,
                tool_ids TEXT
            );

            INSERT INTO Agent_new (
                id, name, description, system_prompt_id, subagent_ids, model, tool_ids
            )
            SELECT
                id,
                name,
                description,
                system_prompt_id,
                CASE
                    WHEN subagent_id IS NULL THEN NULL
                    ELSE json_array(subagent_id)
                END,
                model,
                tool_ids
            FROM Agent;

            DROP TABLE Agent;
            ALTER TABLE Agent_new RENAME TO Agent;
            """
        )
        columns = _agent_table_columns(conn)

    if "subagent_ids" not in columns:
        return

    if not (
        _column_is_not_null(conn, "subagent_ids")
        or _column_is_not_null(conn, "tool_ids")
    ):
        return

    conn.executescript(
        """
        CREATE TABLE Agent_new (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            system_prompt_id INTEGER NOT NULL REFERENCES SystemPrompt(id),
            subagent_ids TEXT,
            model TEXT,
            tool_ids TEXT
        );

        INSERT INTO Agent_new (
            id, name, description, system_prompt_id, subagent_ids, model, tool_ids
        )
        SELECT
            id,
            name,
            description,
            system_prompt_id,
            NULLIF(subagent_ids, '[]'),
            model,
            NULLIF(tool_ids, '[]')
        FROM Agent;

        DROP TABLE Agent;
        ALTER TABLE Agent_new RENAME TO Agent;
        """
    )


def init_agent_db(conn_string: str | None = None) -> None:
    """Create agent tables and seed defaults when empty."""
    conn = _connect(conn_string)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS SystemPrompt (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS Agent (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                system_prompt_id INTEGER NOT NULL REFERENCES SystemPrompt(id),
                subagent_ids TEXT,
                model TEXT,
                tool_ids TEXT
            );
            """
        )
        conn.commit()
        _migrate_agent_table(conn)
        conn.commit()

        count = conn.execute("SELECT COUNT(*) FROM Agent").fetchone()[0]
        if count == 0:
            from db.seed_agents import seed_default_agents

            seed_default_agents(conn)
            conn.commit()
    finally:
        conn.close()


def get_system_prompt(
    prompt_id: int,
    *,
    conn_string: str | None = None,
) -> SystemPromptRow:
    """Load a system prompt row by ID."""
    conn = _connect(conn_string)
    try:
        row = conn.execute(
            "SELECT id, content FROM SystemPrompt WHERE id = ?",
            (prompt_id,),
        ).fetchone()
        if row is None:
            raise AgentNotFoundError(f"Unknown system_prompt_id: {prompt_id}")
        return SystemPromptRow(id=int(row["id"]), content=str(row["content"]))
    finally:
        conn.close()


def get_agent(
    agent_id: int,
    *,
    conn_string: str | None = None,
) -> AgentRow:
    """Load an agent row by ID."""
    conn = _connect(conn_string)
    try:
        row = conn.execute(
            """
            SELECT id, name, description, system_prompt_id, subagent_ids, model, tool_ids
            FROM Agent
            WHERE id = ?
            """,
            (agent_id,),
        ).fetchone()
        if row is None:
            raise AgentNotFoundError(f"Unknown agent_id: {agent_id}")
        return _row_to_agent(row)
    finally:
        conn.close()


def list_agent_ids(*, conn_string: str | None = None) -> list[int]:
    """Return all configured agent IDs."""
    conn = _connect(conn_string)
    try:
        rows = conn.execute("SELECT id FROM Agent ORDER BY id").fetchall()
        return [int(row["id"]) for row in rows]
    finally:
        conn.close()
