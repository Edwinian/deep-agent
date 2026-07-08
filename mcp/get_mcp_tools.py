"""Load MCP tools from configured servers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

_MCP_DIR = Path(__file__).parent
_MATH_SERVER = _MCP_DIR / "math_server.py"
WEATHER_MCP_URL = os.getenv("WEATHER_MCP_URL", "http://127.0.0.1:8001/mcp")


async def get_mcp_tools(token: str | None = None) -> list[BaseTool]:
    """Initialize MCP client and return available tools.

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
            "math_server": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(_MATH_SERVER)],
            },
        }
    )
    return await client.get_tools()
