"""Registry of tool loaders keyed by tool set ID."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.tools import BaseTool

from tools.hotel.get_hotel_tools import HOTEL_TOOLS_ID, get_hotel_tools
from tools.math.get_math_mcp_tools import MATH_MCP_TOOLS_ID, get_math_mcp_tools
from tools.rag.retrieve_tool import RETRIEVE_TOOL_ID, retrieve_tool
from tools.think.think_tool import THINK_TOOL_ID, think_tool
from tools.todo.todo_tools import READ_TODOS_TOOL_ID, read_todos
from tools.weather.get_weather_mcp_tools import WEATHER_MCP_TOOLS_ID, get_weather_mcp_tools
from tools.web_search.web_search_tool import WEB_SEARCH_TOOL_ID, web_search_tool
from utils.retry import wrap_tool_with_retry
from utils.tool_quality_retry import wrap_tool_with_quality_retry

ToolLoader = Callable[[str | None], Awaitable[list[BaseTool]]]

# Network / MCP-backed toolsets: infra backoff + optional semantic quality retry.
# Quality wrap no-ops when the tool args lack a query-like string field.
_RETRYABLE_TOOLSET_IDS = frozenset(
    {
        WEATHER_MCP_TOOLS_ID,
        MATH_MCP_TOOLS_ID,
        HOTEL_TOOLS_ID,
    }
)


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
    WEB_SEARCH_TOOL_ID: _static_tool_loader(web_search_tool),
    RETRIEVE_TOOL_ID: _static_tool_loader(retrieve_tool),
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
        loaded = await loader(token)
        if tool_id in _RETRYABLE_TOOLSET_IDS:
            loaded = [
                wrap_tool_with_quality_retry(wrap_tool_with_retry(tool))
                for tool in loaded
            ]
        tools.extend(loaded)
    return tools
