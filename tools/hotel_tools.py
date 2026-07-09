"""Load hotel booking tools from MCP Toolbox."""

from __future__ import annotations

import os

from langchain_core.tools import BaseTool
from toolbox_langchain import ToolboxClient

HOTEL_TOOLS_ID = 2007
TOOLBOX_URL = os.getenv("TOOLBOX_URL", "http://127.0.0.1:5000")
HOTEL_TOOLSET = os.getenv("TOOLBOX_HOTEL_TOOLSET", "hotel_toolset")

_toolbox_client_instance: ToolboxClient | None = None
_hotel_tools_cache: list[BaseTool] | None = None


def _get_toolbox_client() -> ToolboxClient:
    """Return a process-wide Toolbox client (must stay open for tool calls)."""
    global _toolbox_client_instance
    if _toolbox_client_instance is None:
        _toolbox_client_instance = ToolboxClient(TOOLBOX_URL)
    return _toolbox_client_instance


async def get_hotel_tools(token: str | None = None) -> list[BaseTool]:
    """Load the hotel toolset from a running MCP Toolbox server.

    The Toolbox client session must remain open after load; tool invocations
    reuse its HTTP session.

    Args:
        token: Optional auth token (reserved for future use; not passed to Toolbox).
    """
    _ = token

    global _hotel_tools_cache
    if _hotel_tools_cache is not None:
        return _hotel_tools_cache

    client = _get_toolbox_client()
    _hotel_tools_cache = await client.aload_toolset(HOTEL_TOOLSET)
    return _hotel_tools_cache
