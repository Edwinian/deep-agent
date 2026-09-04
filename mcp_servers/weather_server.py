import os
from urllib.parse import urlparse

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("WeatherTool")


def _access_token_from_context(ctx: Context) -> str | None:
    """Extract a bearer token from the incoming MCP HTTP request."""
    request = ctx.request_context.request
    if request is None:
        return None
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    prefix = "bearer "
    if auth.lower().startswith(prefix):
        return auth[len(prefix) :].strip()
    return auth.strip()


@mcp.tool()
def get_weather(city: str, ctx: Context) -> str:
    """Get current weather for a city."""
    token = _access_token_from_context(ctx)
    if token:
        return f"Sunny in {city} (authorized as {token[:8]}...)"
    return f"Sunny in {city}"


if __name__ == "__main__":
    parsed = urlparse(os.getenv("WEATHER_MCP_URL", "http://127.0.0.1:8001/mcp"))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/mcp"
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.streamable_http_path = path
    mcp.run(transport="streamable-http")
