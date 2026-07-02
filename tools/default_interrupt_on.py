"""Default interrupt-on settings for project tools."""

from langchain.agents.middleware import InterruptOnConfig

from agents.types import InterruptOn
from tools.think_tool import think_tool
from tools.todo_tools import read_todos
from tools.web_search_tool import web_search_tool

_TOOL_PERMISSION_INTERRUPT_ON: InterruptOnConfig = {
    "allowed_decisions": ["approve", "edit", "reject", "respond"],
}

DEFAULT_INTERRUPT_ON: InterruptOn = {
    think_tool.name: False,
    read_todos.name: False,
    web_search_tool.name: True,
}
