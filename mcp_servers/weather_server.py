import os
from urllib.parse import urlparse

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

mcp = FastMCP("WeatherTool")


def _access_token_from_headers(headers: dict[str, str]) -> str | None:
    """Extract a bearer token from incoming MCP HTTP headers."""
    auth = headers.get("authorization") or headers.get("Authorization")
    if not auth:
        return None
    prefix = "bearer "
    if auth.lower().startswith(prefix):
        return auth[len(prefix) :].strip()
    return auth.strip()


@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    headers = get_http_headers()
    token = _access_token_from_headers(headers)
    if token:
        return f"Sunny in {city} (authorized as {token[:8]}...)"
    return f"Sunny in {city}"


if __name__ == "__main__":
    parsed = urlparse(os.getenv("WEATHER_MCP_URL", "http://127.0.0.1:8001/mcp"))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/mcp"
    mcp.run(transport="http", host=host, port=port, path=path)
