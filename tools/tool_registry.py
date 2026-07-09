"""Registry of tool loaders keyed by tool set ID."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.tools import BaseTool

from tools.get_math_mcp_tools import MATH_MCP_TOOLS_ID, get_math_mcp_tools
from tools.get_weather_mcp_tools import WEATHER_MCP_TOOLS_ID, get_weather_mcp_tools
from tools.hotel_tools import HOTEL_TOOLS_ID, get_hotel_tools
from tools.summarize_tool import SUMMARIZE_TOOL_ID, summarize_tool
from tools.think_tool import THINK_TOOL_ID, think_tool
from tools.todo_tools import READ_TODOS_TOOL_ID, read_todos
from tools.web_search_tool import WEB_SEARCH_TOOL_ID, web_search_tool

ToolLoader = Callable[[str | None], Awaitable[list[BaseTool]]]


def _static_tool_loader(tool: BaseTool) -> ToolLoader:
    """Wrap a single tool as an async loader for the registry."""

    async def loader(token: str | None = None) -> list[BaseTool]:
        _ = token
        return [tool]

    return loader


TOOL_REGISTRY: dict[int, ToolLoader] = {
    WEATHER_MCP_TOOLS_ID: get_weather_mcp_tools,
    MATH_MCP_TOOLS_ID: get_math_mcp_tools,
    HOTEL_TOOLS_ID: get_hotel_tools,
    THINK_TOOL_ID: _static_tool_loader(think_tool),
    READ_TODOS_TOOL_ID: _static_tool_loader(read_todos),
    SUMMARIZE_TOOL_ID: _static_tool_loader(summarize_tool),
    WEB_SEARCH_TOOL_ID: _static_tool_loader(web_search_tool),
}


async def resolve_tools(
    tool_ids: list[int],
    token: str | None = None,
) -> list[BaseTool]:
    """Load and merge tools for the requested tool set IDs."""
    tools: list[BaseTool] = []
    for tool_id in tool_ids:
        loader = TOOL_REGISTRY.get(tool_id)
        if loader is None:
            raise KeyError(f"Unknown tool_id: {tool_id}")
        tools.extend(await loader(token))
    return tools
