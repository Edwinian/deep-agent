"""Default interrupt-on settings for project tools."""

from agents.types import InterruptOn
from tools.summarize.summarize_tool import summarize_tool
from tools.think.think_tool import think_tool
from tools.todo.todo_tools import read_todos
from tools.web_search.web_search_tool import web_search_tool

DEFAULT_INTERRUPT_ON: InterruptOn = {
    summarize_tool.name: False,
    think_tool.name: False,
    read_todos.name: False,
    web_search_tool.name: True,
    "get_weather": False,
    "add": False,
    "multiply": False,
    "search-hotels-by-location": False,
    "book-hotel": True,
}
