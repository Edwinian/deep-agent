"""Agent guardrail middleware."""

from langchain.agents.middleware.types import AgentMiddleware

from guardrails.pii import PII_MIDDLEWARE, RedactedPIIResponseMiddleware
from guardrails.tool_call_args_repair import ToolCallArgsRepairMiddleware

GUARDRAILS: list[AgentMiddleware] = [
    *PII_MIDDLEWARE,
    RedactedPIIResponseMiddleware(),
    ToolCallArgsRepairMiddleware(),
]

__all__ = [
    "GUARDRAILS",
    "PII_MIDDLEWARE",
    "RedactedPIIResponseMiddleware",
    "ToolCallArgsRepairMiddleware",
]
