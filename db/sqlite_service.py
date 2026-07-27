"""Shared SQLite connection helper."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SqliteService:
    """Thin wrapper around a SQLite database path / connection string."""

    def __init__(self, conn_string: str) -> None:
        if not conn_string or not str(conn_string).strip():
            raise ValueError("conn_string cannot be empty")
        self.conn_string = str(conn_string)

    def ensure_parent_dir(self) -> None:
        """Create the parent directory for file-backed SQLite DBs."""
        db_path = Path(self.conn_string)
        if db_path.suffix:
            db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self, *, row_factory: bool = True) -> sqlite3.Connection:
        """Open a new connection to ``conn_string``."""
        self.ensure_parent_dir()
        conn = sqlite3.connect(self.conn_string, check_same_thread=False)
        if row_factory:
            conn.row_factory = sqlite3.Row
        return conn

    def execute(
        self,
        sql: str,
        params: tuple | list | None = None,
    ) -> list[sqlite3.Row]:
        """Run a statement and return all rows (connection is closed after)."""
        conn = self.connect()
        try:
            cursor = conn.execute(sql, params or ())
            rows = cursor.fetchall()
            conn.commit()
            return list(rows)
        finally:
            conn.close()

    def executemany(self, sql: str, params_seq: list | tuple) -> None:
        """Run ``executemany`` and commit (connection is closed after)."""
        conn = self.connect()
        try:
            conn.executemany(sql, params_seq)
            conn.commit()
        finally:
            conn.close()

    def executescript(self, script: str) -> None:
        """Run a SQL script and commit (connection is closed after)."""
        conn = self.connect()
        try:
            conn.executescript(script)
            conn.commit()
        finally:
            conn.close()

    def list_tables(self, *, exclude: set[str] | None = None) -> list[str]:
        """Return user table names, excluding SQLite internals and ``exclude``."""
        excluded = exclude or set()
        conn = self.connect()
        try:
            rows = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
            return [
                str(row["name"])
                for row in rows
                if str(row["name"]) not in excluded
            ]
        finally:
            conn.close()

    def table_columns(self, table_name: str) -> set[str]:
        """Return column names for ``table_name`` via ``PRAGMA table_info``."""
        # Identifiers cannot be bound as parameters; validate lightly.
        if not table_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {table_name}")
        conn = self.connect()
        try:
            rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            return {str(row["name"]) for row in rows}
        finally:
            conn.close()

    def __enter__(self) -> sqlite3.Connection:
        self._ctx_conn = self.connect()
        return self._ctx_conn

    def __exit__(self, exc_type, exc, tb) -> None:
        conn = getattr(self, "_ctx_conn", None)
        if conn is None:
            return
        try:
            if exc is None:
                conn.commit()
            else:
                conn.rollback()
        finally:
            conn.close()
            self._ctx_conn = None
