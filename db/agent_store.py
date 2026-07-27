"""SQLite persistence for agent configuration."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from db.sqlite_service import SqliteService

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DB_CONN_STRING = os.getenv(
    "AGENT_DB_CONN_STRING",
    str(_PROJECT_ROOT / "data" / "agents.db"),
)


class AgentNotFoundError(LookupError):
    """Raised when no agent row exists for the requested ID."""


class SkillNotFoundError(LookupError):
    """Raised when no skill row exists for the requested ID."""


@dataclass(frozen=True)
class BaseRow:
    id: int
    created_at: str
    updated_at: str | None
    deleted_at: str | None


@dataclass(frozen=True)
class SystemPromptRow(BaseRow):
    name: str
    content: str


@dataclass(frozen=True)
class SkillRow(BaseRow):
    name: str
    description: str
    content: str


@dataclass(frozen=True)
class AgentRow(BaseRow):
    name: str
    description: str
    system_prompt_id: int
    subagent_ids: list[int] | None
    model: str | None
    tool_ids: list[int] | None
    skill_ids: list[int] | None


def _sqlite(conn_string: str | None = None) -> SqliteService:
    return SqliteService(conn_string or AGENT_DB_CONN_STRING)


def _connect(conn_string: str | None = None) -> sqlite3.Connection:
    """Open a SQLite connection via :class:`SqliteService`."""
    return _sqlite(conn_string).connect()


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


def _base_row_fields(row: sqlite3.Row) -> dict[str, int | str | None]:
    return {
        "id": int(row["id"]),
        "created_at": str(row["created_at"]),
        "updated_at": (
            str(row["updated_at"]) if row["updated_at"] is not None else None
        ),
        "deleted_at": (
            str(row["deleted_at"]) if row["deleted_at"] is not None else None
        ),
    }


def _row_to_agent(row: sqlite3.Row) -> AgentRow:
    return AgentRow(
        **_base_row_fields(row),
        name=str(row["name"]),
        description=str(row["description"]),
        system_prompt_id=int(row["system_prompt_id"]),
        subagent_ids=_parse_optional_id_list(
            row["subagent_ids"],
            field_name="subagent_ids",
        ),
        model=str(row["model"]) if row["model"] is not None else None,
        tool_ids=_parse_optional_id_list(row["tool_ids"], field_name="tool_ids"),
        skill_ids=_parse_optional_id_list(row["skill_ids"], field_name="skill_ids"),
    )


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def _agent_table_columns(conn: sqlite3.Connection) -> set[str]:
    return _table_columns(conn, "Agent")


def _column_is_not_null(conn: sqlite3.Connection, column_name: str) -> bool:
    rows = conn.execute("PRAGMA table_info(Agent)").fetchall()
    for row in rows:
        if str(row["name"]) == column_name:
            return bool(row["notnull"])
    return False


def _migrate_timestamp_columns(conn: sqlite3.Connection) -> None:
    """Add created_at, updated_at, and deleted_at to existing tables."""
    for table in ("SystemPrompt", "Skill", "Agent"):
        columns = _table_columns(conn, table)
        if "created_at" not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN created_at TEXT")
            conn.execute(
                f"UPDATE {table} SET created_at = datetime('now') "
                "WHERE created_at IS NULL"
            )
        if "updated_at" not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN updated_at TEXT")
        if "deleted_at" not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN deleted_at TEXT")


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
                tool_ids TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT,
                deleted_at TEXT
            );

            INSERT INTO Agent_new (
                id,
                name,
                description,
                system_prompt_id,
                subagent_ids,
                model,
                tool_ids,
                created_at,
                updated_at,
                deleted_at
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
                tool_ids,
                COALESCE(created_at, datetime('now')),
                updated_at,
                deleted_at
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
            tool_ids TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT,
            deleted_at TEXT
        );

        INSERT INTO Agent_new (
            id,
            name,
            description,
            system_prompt_id,
            subagent_ids,
            model,
            tool_ids,
            created_at,
            updated_at,
            deleted_at
        )
        SELECT
            id,
            name,
            description,
            system_prompt_id,
            NULLIF(subagent_ids, '[]'),
            model,
            NULLIF(tool_ids, '[]'),
            COALESCE(created_at, datetime('now')),
            updated_at,
            deleted_at
        FROM Agent;

        DROP TABLE Agent;
        ALTER TABLE Agent_new RENAME TO Agent;
        """
    )


def _migrate_skill_schema(conn: sqlite3.Connection) -> None:
    """Add Skill table and nullable skill_ids on Agent."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS Skill (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT,
            deleted_at TEXT
        );
        """
    )
    columns = _agent_table_columns(conn)
    if "skill_ids" not in columns:
        conn.execute("ALTER TABLE Agent ADD COLUMN skill_ids TEXT")


def _migrate_system_prompt_name(conn: sqlite3.Connection) -> None:
    """Add name to SystemPrompt for existing databases."""
    from constants.agent_name import AgentName

    columns = _table_columns(conn, "SystemPrompt")
    if "name" not in columns:
        conn.execute("ALTER TABLE SystemPrompt ADD COLUMN name TEXT")
        conn.execute(
            """
            UPDATE SystemPrompt
            SET name = CASE
                WHEN id = 1 THEN ?
                WHEN id = 2 THEN ?
                ELSE 'system-prompt-' || id
            END
            WHERE name IS NULL
            """,
            (AgentName.RESEARCH_AGENT, AgentName.GENERAL_AGENT),
        )


def _migrate_system_prompt_default_names(conn: sqlite3.Connection) -> None:
    """Rename legacy default SystemPrompt names to agent-aligned values."""
    from constants.agent_name import AgentName

    conn.execute(
        """
        UPDATE SystemPrompt
        SET name = ?
        WHERE id = 1
          AND name IN ('system-prompt-1', 'research-system-prompt')
        """,
        (AgentName.RESEARCH_AGENT,),
    )
    conn.execute(
        """
        UPDATE SystemPrompt
        SET name = ?
        WHERE id = 2
          AND name IN ('system-prompt-2', 'general-system-prompt')
        """,
        (AgentName.GENERAL_AGENT,),
    )


def _migrate_rag_agent(conn: sqlite3.Connection) -> None:
    """Add rag_agent and attach it to general_agent for existing databases."""
    from datetime import datetime, timezone

    from agents.ids import GENERAL_AGENT_ID, RAG_AGENT_ID
    from constants.agent_name import AgentName
    from constants.model_name import ModelName
    from db.seed_agents import RAG_AGENT_DESCRIPTION, _build_rag_system_prompt

    RAG_SYSTEM_PROMPT_ID = 3
    RETRIEVE_TOOL_ID = 2008

    existing = conn.execute(
        "SELECT id FROM Agent WHERE id = ? OR name = ?",
        (RAG_AGENT_ID, AgentName.RAG_AGENT),
    ).fetchone()
    if existing is None:
        now = datetime.now(timezone.utc).isoformat()
        rag_prompt = _build_rag_system_prompt()
        conn.execute(
            """
            INSERT INTO SystemPrompt (id, name, content, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                content = excluded.content
            """,
            (RAG_SYSTEM_PROMPT_ID, AgentName.RAG_AGENT, rag_prompt, now),
        )
        conn.execute(
            """
            INSERT INTO Agent (
                id,
                name,
                description,
                system_prompt_id,
                subagent_ids,
                model,
                tool_ids,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                RAG_AGENT_ID,
                AgentName.RAG_AGENT,
                RAG_AGENT_DESCRIPTION,
                RAG_SYSTEM_PROMPT_ID,
                None,
                ModelName.GROK_4_3.with_provider(),
                json.dumps([RETRIEVE_TOOL_ID]),
                now,
            ),
        )

    general_row = conn.execute(
        "SELECT subagent_ids FROM Agent WHERE id = ?",
        (GENERAL_AGENT_ID,),
    ).fetchone()
    if general_row is None:
        return

    subagent_ids = _parse_optional_id_list(
        general_row["subagent_ids"],
        field_name="subagent_ids",
    ) or []
    if RAG_AGENT_ID not in subagent_ids:
        subagent_ids.append(RAG_AGENT_ID)
        conn.execute(
            "UPDATE Agent SET subagent_ids = ? WHERE id = ?",
            (json.dumps(subagent_ids), GENERAL_AGENT_ID),
        )


def _migrate_rag_agent_description(conn: sqlite3.Connection) -> None:
    """Refresh rag_agent description shown in the task tool's available agents list."""
    from datetime import datetime, timezone

    from agents.ids import RAG_AGENT_ID
    from db.seed_agents import RAG_AGENT_DESCRIPTION

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        UPDATE Agent
        SET description = ?, updated_at = ?
        WHERE id = ?
        """,
        (RAG_AGENT_DESCRIPTION, now, RAG_AGENT_ID),
    )


def _migrate_agent_names(conn: sqlite3.Connection) -> None:
    """Rename legacy kebab-case agent and system-prompt names to snake_case."""
    from constants.agent_name import AgentName

    legacy_names = {
        "research-agent": AgentName.RESEARCH_AGENT,
        "general-agent": AgentName.GENERAL_AGENT,
        "rag-agent": AgentName.RAG_AGENT,
    }
    for old_name, new_name in legacy_names.items():
        conn.execute(
            "UPDATE SystemPrompt SET name = ? WHERE name = ?",
            (new_name, old_name),
        )
        conn.execute(
            "UPDATE Agent SET name = ? WHERE name = ?",
            (new_name, old_name),
        )


def _migrate_function_prompt_names(conn: sqlite3.Connection) -> None:
    """Rename legacy kebab-case RAG function prompt names to snake_case."""
    from constants.function_name import FunctionName

    legacy_names = {
        "generate-answer": FunctionName.GENERATE_ANSWER,
        "grade-documents": FunctionName.GRADE_DOCUMENTS,
        "rewrite-query": FunctionName.REWRITE_QUERY,
    }
    for old_name, new_name in legacy_names.items():
        conn.execute(
            "UPDATE SystemPrompt SET name = ? WHERE name = ?",
            (new_name, old_name),
        )


def _migrate_generate_answer_prompt(conn: sqlite3.Connection) -> None:
    """Add generate_answer system prompt for existing databases."""
    from datetime import datetime, timezone

    from constants.function_name import FunctionName
    from prompts.generate_answer import GENERATE_ANSWER

    GENERATE_ANSWER_SYSTEM_PROMPT_ID = 4

    existing = conn.execute(
        "SELECT id FROM SystemPrompt WHERE id = ? OR name = ?",
        (GENERATE_ANSWER_SYSTEM_PROMPT_ID, FunctionName.GENERATE_ANSWER),
    ).fetchone()
    if existing is not None:
        return

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO SystemPrompt (id, name, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (GENERATE_ANSWER_SYSTEM_PROMPT_ID, FunctionName.GENERATE_ANSWER, GENERATE_ANSWER, now),
    )


def _migrate_rag_tool_prompts(conn: sqlite3.Connection) -> None:
    """Add RAG tool system prompts for existing databases."""
    from datetime import datetime, timezone

    from constants.function_name import FunctionName
    from prompts.generate_answer import GENERATE_ANSWER
    from prompts.grade_documents import GRADE_DOCUMENTS
    from prompts.rewrite_query import REWRITE_QUERY

    prompts = (
        (4, FunctionName.GENERATE_ANSWER, GENERATE_ANSWER),
        (5, FunctionName.GRADE_DOCUMENTS, GRADE_DOCUMENTS),
        (6, FunctionName.REWRITE_QUERY, REWRITE_QUERY),
    )
    now = datetime.now(timezone.utc).isoformat()
    for prompt_id, name, content in prompts:
        existing = conn.execute(
            "SELECT id FROM SystemPrompt WHERE id = ? OR name = ?",
            (prompt_id, name),
        ).fetchone()
        if existing is not None:
            continue
        conn.execute(
            """
            INSERT INTO SystemPrompt (id, name, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (prompt_id, name, content, now),
        )


def _migrate_general_agent_prompt(conn: sqlite3.Connection) -> None:
    """Refresh agent system prompts (delegation rules + privacy guardrails)."""
    from datetime import datetime, timezone

    from db.seed_agents import (
        GENERAL_SYSTEM_PROMPT_ID,
        RAG_SYSTEM_PROMPT_ID,
        RESEARCH_SYSTEM_PROMPT_ID,
        _build_general_system_prompt,
        _build_rag_system_prompt,
        _build_research_system_prompt,
    )

    now = datetime.now(timezone.utc).isoformat()
    prompts = (
        (RESEARCH_SYSTEM_PROMPT_ID, _build_research_system_prompt()),
        (GENERAL_SYSTEM_PROMPT_ID, _build_general_system_prompt()),
        (RAG_SYSTEM_PROMPT_ID, _build_rag_system_prompt()),
    )
    for prompt_id, content in prompts:
        conn.execute(
            """
            UPDATE SystemPrompt
            SET content = ?, updated_at = ?
            WHERE id = ?
            """,
            (content, now, prompt_id),
        )


def init_agent_db(conn_string: str | None = None) -> None:
    """Create agent tables and seed defaults when empty."""
    db = _sqlite(conn_string)
    with db as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS SystemPrompt (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT,
                deleted_at TEXT
            );

            CREATE TABLE IF NOT EXISTS Skill (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT,
                deleted_at TEXT
            );

            CREATE TABLE IF NOT EXISTS Agent (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                system_prompt_id INTEGER NOT NULL REFERENCES SystemPrompt(id),
                subagent_ids TEXT,
                model TEXT,
                tool_ids TEXT,
                skill_ids TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT,
                deleted_at TEXT
            );
            """
        )
        _migrate_timestamp_columns(conn)
        _migrate_agent_table(conn)
        _migrate_skill_schema(conn)
        _migrate_system_prompt_name(conn)
        _migrate_system_prompt_default_names(conn)
        _migrate_agent_names(conn)
        _migrate_rag_agent(conn)
        _migrate_function_prompt_names(conn)
        _migrate_generate_answer_prompt(conn)
        _migrate_rag_tool_prompts(conn)
        _migrate_general_agent_prompt(conn)
        _migrate_rag_agent_description(conn)

        count = conn.execute("SELECT COUNT(*) FROM Agent").fetchone()[0]
        needs_seed = count == 0

    if needs_seed:
        from db.seed_agents import seed_default_agents

        seed_default_agents(db)

    # Migrate the seeded default model for existing databases.
    #
    # Older versions used a reasoning-oriented Grok model which can emit
    # incomplete tool-call arguments (e.g. `{}`), causing tool validation
    # failures. Keep existing DBs working without manual intervention.
    from datetime import datetime, timezone

    from agents.ids import GENERAL_AGENT_ID, RAG_AGENT_ID, RESEARCH_AGENT_ID
    from constants.model_name import ModelName

    now = datetime.now(timezone.utc).isoformat()
    target_model = ModelName.GROK_4_FAST_NON_REASONING.with_provider()
    legacy_model = ModelName.GROK_4_3.with_provider()

    with db as conn:
        conn.execute(
            """
            UPDATE Agent
            SET model = ?, updated_at = ?
            WHERE id IN (?, ?, ?)
              AND model = ?
            """,
            (
                target_model,
                now,
                RESEARCH_AGENT_ID,
                GENERAL_AGENT_ID,
                RAG_AGENT_ID,
                legacy_model,
            ),
        )


def get_system_prompt(
    prompt_id: int,
    *,
    conn_string: str | None = None,
) -> SystemPromptRow:
    """Load a system prompt row by ID."""
    conn = _connect(conn_string)
    try:
        row = conn.execute(
            """
            SELECT id, name, content, created_at, updated_at, deleted_at
            FROM SystemPrompt
            WHERE id = ?
            """,
            (prompt_id,),
        ).fetchone()
        if row is None:
            raise AgentNotFoundError(f"Unknown system_prompt_id: {prompt_id}")
        return SystemPromptRow(
            **_base_row_fields(row),
            name=str(row["name"]),
            content=str(row["content"]),
        )
    finally:
        conn.close()


def get_system_prompt_by_name(
    name: str,
    *,
    conn_string: str | None = None,
) -> SystemPromptRow:
    """Load a system prompt row by name."""
    conn = _connect(conn_string)
    try:
        row = conn.execute(
            """
            SELECT id, name, content, created_at, updated_at, deleted_at
            FROM SystemPrompt
            WHERE name = ?
            """,
            (str(name),),
        ).fetchone()
        if row is None:
            raise AgentNotFoundError(f"Unknown system prompt name: {name}")
        return SystemPromptRow(
            **_base_row_fields(row),
            name=str(row["name"]),
            content=str(row["content"]),
        )
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
            SELECT
                id,
                name,
                description,
                system_prompt_id,
                subagent_ids,
                model,
                tool_ids,
                skill_ids,
                created_at,
                updated_at,
                deleted_at
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


def get_skill(
    skill_id: int,
    *,
    conn_string: str | None = None,
) -> SkillRow:
    """Load a skill row by ID."""
    conn = _connect(conn_string)
    try:
        row = conn.execute(
            """
            SELECT id, name, description, content, created_at, updated_at, deleted_at
            FROM Skill
            WHERE id = ?
            """,
            (skill_id,),
        ).fetchone()
        if row is None:
            raise SkillNotFoundError(f"Unknown skill_id: {skill_id}")
        return SkillRow(
            **_base_row_fields(row),
            name=str(row["name"]),
            description=str(row["description"]),
            content=str(row["content"]),
        )
    finally:
        conn.close()


def get_skills(
    skill_ids: list[int],
    *,
    conn_string: str | None = None,
) -> list[SkillRow]:
    """Load skill rows for the given IDs, preserving order."""
    return [get_skill(skill_id, conn_string=conn_string) for skill_id in skill_ids]


def list_skill_ids(*, conn_string: str | None = None) -> list[int]:
    """Return all configured skill IDs."""
    conn = _connect(conn_string)
    try:
        rows = conn.execute("SELECT id FROM Skill ORDER BY id").fetchall()
        return [int(row["id"]) for row in rows]
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
