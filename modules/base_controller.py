"""Abstract base for feature-module controllers.

Every feature module exposes a ``Controller`` subclass that:
  * declares its URL ``PREFIX`` (e.g. ``"/agents"``),
  * builds its routes inside :meth:`_register_routes`.

The base class wires the common boilerplate: it constructs an
:class:`fastapi.APIRouter` tagged with the module name, assigns it to
``self.router``, and calls :meth:`_register_routes` once on init.

The OpenAPI ``tags`` value defaults to the prefix with the leading slash
stripped (``"/system-prompts"`` -> ``"system-prompts"``); subclasses can
override :attr:`TAG` for custom tag names.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fastapi import APIRouter


class BaseController(ABC):
    """Mounts a feature router at :attr:`PREFIX` and registers its routes.

    Subclasses must set :attr:`PREFIX` and implement :meth:`_register_routes`.
    The router is built from ``PREFIX`` (and a derived tag) before routes are
    registered, so subclasses can reference ``self.router`` directly.
    """

    PREFIX: str = ""
    TAG: str | None = None

    router: APIRouter

    def __init__(self) -> None:
        tag = self.TAG or self.PREFIX.lstrip("/")
        self.router = APIRouter(prefix=self.PREFIX, tags=[tag])
        self._register_routes()

    @abstractmethod
    def _register_routes(self) -> None:
        """Register all routes onto :attr:`self.router`."""
