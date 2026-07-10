"""Pass access tokens from agent invocations to MCP HTTP servers."""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable
from contextlib import contextmanager

from langchain_mcp_adapters.interceptors import MCPToolCallRequest, ToolCallInterceptor
from mcp.types import CallToolResult

mcp_access_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mcp_access_token",
    default=None,
)


def parse_authorization_header(authorization: str | None) -> str | None:
    """Extract a bearer token from an HTTP Authorization header value."""
    if not authorization:
        return None
    prefix = "bearer "
    if authorization.lower().startswith(prefix):
        return authorization[len(prefix) :].strip() or None
    return authorization.strip() or None


def authorization_headers(token: str | None = None) -> dict[str, str]:
    """Build an Authorization header for MCP HTTP transports."""
    resolved = token if token is not None else mcp_access_token.get()
    if not resolved:
        return {}
    if resolved.lower().startswith("bearer "):
        return {"Authorization": resolved}
    return {"Authorization": f"Bearer {resolved}"}


@contextmanager
def mcp_access_token_context(token: str | None):
    """Bind the caller access token for the current invoke/stream turn."""
    reset = mcp_access_token.set(token)
    try:
        yield
    finally:
        mcp_access_token.reset(reset)


class McpAuthInterceptor:
    """Inject the current access token into each MCP tool HTTP request."""

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[CallToolResult]],
    ) -> CallToolResult:
        auth_headers = authorization_headers()
        if not auth_headers:
            return await handler(request)
        merged = dict(request.headers or {})
        merged.update(auth_headers)
        return await handler(request.override(headers=merged))


# Satisfy ToolCallInterceptor structural typing for MultiServerMCPClient.
MCP_AUTH_INTERCEPTOR: ToolCallInterceptor = McpAuthInterceptor()
