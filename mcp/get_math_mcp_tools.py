"""Load MCP tools from the math stdio server."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

_MCP_DIR = Path(__file__).parent
_MATH_SERVER = _MCP_DIR / "math_server.py"


async def get_math_mcp_tools(token: str | None = None) -> list[BaseTool]:
    """Initialize the math MCP client and return its tools.

    Args:
        token: Optional auth token (reserved for future use; not passed to servers).
    """
    _ = token

    client = MultiServerMCPClient(
        {
            "math_server": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(_MATH_SERVER)],
            },
        }
    )
    return await client.get_tools()
