"""Default interrupt-on settings for project tools."""

from agents.types import InterruptOn
from tools.think_tool import think_tool
from tools.todo_tools import read_todos
from tools.web_search_tool import web_search_tool

DEFAULT_INTERRUPT_ON: InterruptOn = {
    think_tool.name: False,
    read_todos.name: False,
    web_search_tool.name: False,
}
