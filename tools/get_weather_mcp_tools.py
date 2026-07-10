"""Load MCP tools from the weather HTTP server."""

from __future__ import annotations

import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from tools.mcp_auth import MCP_AUTH_INTERCEPTOR, authorization_headers

WEATHER_MCP_TOOLS_ID = 2001
WEATHER_MCP_URL = os.getenv("WEATHER_MCP_URL", "http://127.0.0.1:8001/mcp")


async def get_weather_mcp_tools(token: str | None = None) -> list[BaseTool]:
    """Initialize the weather MCP client and return its tools.

    Args:
        token: Optional access token sent as ``Authorization: Bearer`` on MCP
            HTTP requests. Per-invoke tokens are also read from
            ``tools.mcp_auth.mcp_access_token`` when agents are cached.
    """
    connection: dict[str, object] = {
        "transport": "http",
        "url": WEATHER_MCP_URL,
    }
    headers = authorization_headers(token)
    if headers:
        connection["headers"] = headers

    client = MultiServerMCPClient(
        {"weather_server": connection},
        tool_interceptors=[MCP_AUTH_INTERCEPTOR],
    )
    return await client.get_tools()
