"""Default interrupt-on settings for project tools."""

from agents.types import InterruptOn
from tools.rag.retrieve_tool import retrieve_tool
from tools.think.think_tool import think_tool
from tools.todo.todo_tools import read_todos
from tools.web_search.web_search_tool import web_search_tool

DEFAULT_INTERRUPT_ON: InterruptOn = {
    think_tool.name: False,
    read_todos.name: False,
    web_search_tool.name: True,
    retrieve_tool.name: False,
    "get_weather": False,
    "add": False,
    "multiply": False,
    "search-hotels-by-location": False,
    "book-hotel": True,
}
