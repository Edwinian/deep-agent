"""Factory for LangGraph checkpointers."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import TypeVar

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQLITE_CONN_STRING = os.getenv(
    "SQLITE_CONN_STRING",
    str(_PROJECT_ROOT / "data" / "checkpoints.db"),
)
MOCK_POSTGRES_CONN_STRING = "postgresql://user:password@localhost:5432/deep_agents"

_T = TypeVar("_T", bound=BaseCheckpointSaver)


class CheckpointerType(str, Enum):
    """Supported checkpointer backends per LangGraph checkpoint savers."""

    IN_MEMORY = "in_memory"
    ASYNC_SQLITE = "async_sqlite"
    ASYNC_POSTGRESQL = "async_postgresql"


def _ensure_sqlite_parent_dir(conn_string: str) -> None:
    """Create the parent directory for a SQLite file path when applicable."""
    db_path = Path(conn_string)
    if db_path.suffix:
        db_path.parent.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def async_sqlite_checkpointer(
    conn_string: str = SQLITE_CONN_STRING,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Open an async SQLite checkpointer with tables initialized.

    Usage::

        async with async_sqlite_checkpointer() as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
    """
    _ensure_sqlite_parent_dir(conn_string)
    async with AsyncSqliteSaver.from_conn_string(conn_string) as checkpointer:
        await checkpointer.setup()
        yield checkpointer


@asynccontextmanager
async def async_postgres_checkpointer(
    conn_string: str = MOCK_POSTGRES_CONN_STRING,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Open an async Postgres checkpointer with tables initialized.

    Usage::

        async with async_postgres_checkpointer() as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
    """
    async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
        await checkpointer.setup()
        yield checkpointer


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "Cannot create an async checkpointer while an event loop is running. "
        "Use AsyncSqliteSaver.from_conn_string() or "
        "AsyncPostgresSaver.from_conn_string() directly in async code."
    )


async def _enter_async_context(
    context: AbstractAsyncContextManager[_T],
) -> _T:
    return await context.__aenter__()


def _open_async_checkpointer(
    context: AbstractAsyncContextManager[_T],
) -> _T:
    return _run_async(_enter_async_context(context))


_MEMORY_SAVER: MemorySaver | None = None
_SQLITE_SAVER: AsyncSqliteSaver | None = None
_sqlite_exit_stack: AsyncExitStack | None = None


async def init_sqlite_checkpointer(
    conn_string: str | None = None,
) -> AsyncSqliteSaver:
    """Initialize the process-wide async SQLite checkpointer.

    Call from the FastAPI lifespan before compiling agents under uvicorn.
    """
    global _SQLITE_SAVER, _sqlite_exit_stack
    if _SQLITE_SAVER is not None:
        return _SQLITE_SAVER

    resolved = conn_string or SQLITE_CONN_STRING
    _ensure_sqlite_parent_dir(resolved)
    _sqlite_exit_stack = AsyncExitStack()
    _SQLITE_SAVER = await _sqlite_exit_stack.enter_async_context(
        async_sqlite_checkpointer(resolved)
    )
    return _SQLITE_SAVER


async def get_sqlite_checkpointer() -> AsyncSqliteSaver:
    """Return the initialized process-wide async SQLite checkpointer."""
    if _SQLITE_SAVER is None:
        raise RuntimeError(
            "SQLite checkpointer is not initialized. Call init_sqlite_checkpointer() "
            "from the FastAPI lifespan before handling requests."
        )
    return _SQLITE_SAVER


async def delete_thread_checkpoints(thread_id: str) -> None:
    """Delete all LangGraph checkpoints and writes for a thread."""
    checkpointer = await get_sqlite_checkpointer()
    await checkpointer.adelete_thread(thread_id)


async def close_sqlite_checkpointer() -> None:
    """Close the process-wide async SQLite checkpointer."""
    global _SQLITE_SAVER, _sqlite_exit_stack
    if _sqlite_exit_stack is None:
        return
    await _sqlite_exit_stack.aclose()
    _sqlite_exit_stack = None
    _SQLITE_SAVER = None


def get_checkpointer(checkpointer_type: CheckpointerType) -> BaseCheckpointSaver:
    """Return a checkpointer for the given backend type.

    ``IN_MEMORY`` and ``ASYNC_SQLITE`` return process-wide singletons so thread
    checkpoints survive across repeated ``compile_agent`` calls in the same
    server process.

    For async savers under uvicorn, call ``init_sqlite_checkpointer()`` from
    the application lifespan before the first agent compile.
  """
    global _MEMORY_SAVER, _SQLITE_SAVER
    if checkpointer_type is CheckpointerType.IN_MEMORY:
        if _MEMORY_SAVER is None:
            _MEMORY_SAVER = MemorySaver()
        return _MEMORY_SAVER

    if checkpointer_type is CheckpointerType.ASYNC_SQLITE:
        if _SQLITE_SAVER is not None:
            return _SQLITE_SAVER
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            _SQLITE_SAVER = _open_async_checkpointer(
                async_sqlite_checkpointer(SQLITE_CONN_STRING)
            )
            return _SQLITE_SAVER
        raise RuntimeError(
            "SQLite checkpointer is not initialized. Call init_sqlite_checkpointer() "
            "from the FastAPI lifespan before handling requests."
        )

    if checkpointer_type is CheckpointerType.ASYNC_POSTGRESQL:
        return _open_async_checkpointer(async_postgres_checkpointer())

    raise ValueError(f"Unsupported checkpointer type: {checkpointer_type}")
