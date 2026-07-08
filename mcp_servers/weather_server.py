import os
from urllib.parse import urlparse

from fastmcp import FastMCP

mcp = FastMCP("WeatherTool")

@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny in {city}"


if __name__ == "__main__":
    parsed = urlparse(os.getenv("WEATHER_MCP_URL", "http://127.0.0.1:8001/mcp"))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/mcp"
    mcp.run(transport="http", host=host, port=port, path=path)
