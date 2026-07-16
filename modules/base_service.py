"""Abstract CRUD service used by feature modules.

Concrete services bind :class:`BaseService` to a specific table + row type
and implement the four core operations: ``get_many``, ``create``, ``update``,
``delete``. Services are async to stay compatible with FastAPI path operations
even when the underlying store is synchronous (the default SQLite store here
opens one short-lived connection per call, which is safe to wrap in a thread).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

RowT = TypeVar("RowT")
CreateT = TypeVar("CreateT")
UpdateT = TypeVar("UpdateT")


class BaseService(ABC, Generic[RowT, CreateT, UpdateT]):
    """Generic CRUD contract over a single table.

    Type parameters:
        RowT: frozen dataclass / pydantic model returned by reads.
        CreateT: payload accepted by :meth:`create`.
        UpdateT: payload accepted by :meth:`update` (partial fields allowed).
    """

    @abstractmethod
    async def get_many(self) -> list[RowT]:
        """Return all non-deleted rows ordered by id."""

    @abstractmethod
    async def get_one(self, row_id: int) -> RowT:
        """Return a single row by id, raising ``LookupError`` if missing."""

    @abstractmethod
    async def create(self, payload: CreateT) -> RowT:
        """Insert a new row and return the stored record."""

    @abstractmethod
    async def update(self, row_id: int, payload: UpdateT) -> RowT:
        """Patch an existing row with the non-null fields of ``payload``."""

    @abstractmethod
    async def delete(self, row_id: int) -> None:
        """Soft-delete a row by setting ``deleted_at``."""
