"""Load MCP tools from the weather HTTP server."""

from __future__ import annotations

import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

WEATHER_MCP_TOOLS_ID = 2001
WEATHER_MCP_URL = os.getenv("WEATHER_MCP_URL", "http://127.0.0.1:8001/mcp")


async def get_weather_mcp_tools(token: str | None = None) -> list[BaseTool]:
    """Initialize the weather MCP client and return its tools.

    Args:
        token: Optional auth token (reserved for future use; not passed to servers).
    """
    _ = token

    client = MultiServerMCPClient(
        {
            "weather_server": {
                "transport": "http",
                "url": WEATHER_MCP_URL,
            },
        }
    )
    return await client.get_tools()
