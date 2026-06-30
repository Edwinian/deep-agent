"""Default interrupt-on settings for project tools."""

from langchain.agents.middleware import InterruptOnConfig

from deep_agents_from_scratch.agents.types import InterruptOn
from deep_agents_from_scratch.tools.think_tool import think_tool
from deep_agents_from_scratch.tools.todo_tools import read_todos
from deep_agents_from_scratch.tools.web_search_tool import web_search_tool

_DEFAULT_TOOL_INTERRUPT_ON: InterruptOnConfig = {
    "allowed_decisions": ["approve", "edit", "reject"],
}

DEFAULT_INTERRUPT_ON: InterruptOn = {
    think_tool.name: _DEFAULT_TOOL_INTERRUPT_ON,
    read_todos.name: _DEFAULT_TOOL_INTERRUPT_ON,
    web_search_tool.name: _DEFAULT_TOOL_INTERRUPT_ON,
}
