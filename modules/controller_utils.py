"""Shared helpers for module controllers."""

from __future__ import annotations

from fastapi import HTTPException


def row_not_found(exc: BaseException) -> HTTPException:
    """Map a store ``LookupError`` to a 404 ``HTTPException``."""
    return HTTPException(status_code=404, detail=str(exc))
