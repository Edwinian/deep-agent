"""Agent guardrail middleware."""

from langchain.agents.middleware.types import AgentMiddleware

from guardrails.jailbreak import JailbreakMiddleware
from guardrails.moderation import ModerationMiddleware
from guardrails.pii import PII_MIDDLEWARE, RedactedPIIResponseMiddleware
from guardrails.relevance import RelevanceMiddleware
from guardrails.tool_call_args_repair import ToolCallArgsRepairMiddleware

GUARDRAILS: list[AgentMiddleware] = [
    JailbreakMiddleware(),
    ModerationMiddleware(),
    RelevanceMiddleware(),
    *PII_MIDDLEWARE,
    RedactedPIIResponseMiddleware(),
    ToolCallArgsRepairMiddleware(),
]

__all__ = [
    "GUARDRAILS",
    "JailbreakMiddleware",
    "ModerationMiddleware",
    "PII_MIDDLEWARE",
    "RedactedPIIResponseMiddleware",
    "RelevanceMiddleware",
    "ToolCallArgsRepairMiddleware",
]
