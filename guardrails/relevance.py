"""Relevance classifier middleware — keep requests inside the agent's scope."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe
from langgraph.runtime import Runtime
from langsmith import traceable
from pydantic import BaseModel, Field
from typing_extensions import override

from constants.model_name import ModelName
from guardrails.input import jump_to_end, latest_user_text

logger = logging.getLogger(__name__)

RELEVANCE_REFUSAL = (
    "That request is outside what I can help with. I can help with web research, "
    "indexed documents, weather, math, and hotel search or booking."
)

# Scope of *this* product. A customer-service agent would use a narrower
# description, which is why "How tall is the Empire State Building?" is
# off-topic in the OpenAI example but on-topic here (web research is in scope).
AGENT_INTENDED_SCOPE = """\
This agent helps with:
- Web research and live or factual questions
- RAG over indexed documents (vector DB, Lilian Weng posts, stored docs)
- Weather lookups
- Math (add, multiply)
- Hotel search and booking
- Planning those tasks with todos and files

Greetings, clarifications, and follow-ups about an in-scope task are relevant.
"""

CLASSIFIER_SYSTEM_PROMPT = f"""\
You are a relevance classifier for an AI agent.

Ensure requests stay within the intended scope by flagging off-topic queries.

Intended scope:
{AGENT_INTENDED_SCOPE}

Mark relevant=false when the request cannot be served by those capabilities \
(for example personal companionship, therapy, or asking the agent to act as \
an unrelated product).

Mark relevant=true for in-scope work, including factual questions such as \
"How tall is the Empire State Building?" — web research is in scope here.

Return only the structured verdict. Do not answer the user's request.
"""

_DEFAULT_CLASSIFIER_MODEL = ModelName.GROK_4_FAST_NON_REASONING.with_provider()


class RelevanceVerdict(BaseModel):
    """Structured classifier output."""

    relevant: bool = Field(
        description="True if the user request is inside the agent's intended scope."
    )
    reason: str = Field(
        default="",
        description="Short explanation of why the request is relevant or off-topic.",
    )


class RelevanceMiddleware(AgentMiddleware):
    """Block off-topic queries before the agent runs.

    Uses a small classifier model on the latest human message (``before_agent``).
    HITL resumes and tool-loop continues skip classification because the last
    message is not a new user turn.
    """

    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        self._model_name = model or _DEFAULT_CLASSIFIER_MODEL
        self._model = None

    def _classifier(self):
        if self._model is None:
            self._model = init_chat_model(model=self._model_name, temperature=0)
        return self._model

    @override
    @hook_config(can_jump_to=["end"])
    def before_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        text = latest_user_text(list(state.get("messages") or []))
        if text is None:
            return None
        try:
            verdict = classify_relevance(text, model=self._classifier())
        except Exception:
            logger.exception("Relevance classifier failed; allowing the request")
            return None
        if not verdict.relevant:
            logger.warning("Blocked off-topic query: %s", verdict.reason)
            return jump_to_end(RELEVANCE_REFUSAL)
        return None

    @override
    @hook_config(can_jump_to=["end"])
    async def abefore_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        text = latest_user_text(list(state.get("messages") or []))
        if text is None:
            return None
        try:
            verdict = await aclassify_relevance(text, model=self._classifier())
        except Exception:
            logger.exception("Relevance classifier failed; allowing the request")
            return None
        if not verdict.relevant:
            logger.warning("Blocked off-topic query: %s", verdict.reason)
            return jump_to_end(RELEVANCE_REFUSAL)
        return None


def _classifier_messages(text: str) -> list[Any]:
    return [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=text),
    ]


def _parse_verdict(result: Any) -> RelevanceVerdict:
    if isinstance(result, RelevanceVerdict):
        return result
    if isinstance(result, dict):
        return RelevanceVerdict.model_validate(result)
    return RelevanceVerdict.model_validate(
        {"relevant": True, "reason": "Classifier returned an unexpected payload."}
    )


@observe(name="classify_relevance")
@traceable(name="classify_relevance")
def classify_relevance(text: str, *, model: Any) -> RelevanceVerdict:
    """Return a structured relevance verdict for ``text``."""
    result = model.with_structured_output(RelevanceVerdict).invoke(
        _classifier_messages(text)
    )
    return _parse_verdict(result)


@observe(name="classify_relevance")
@traceable(name="classify_relevance")
async def aclassify_relevance(text: str, *, model: Any) -> RelevanceVerdict:
    """Async structured relevance verdict for ``text``."""
    result = await model.with_structured_output(RelevanceVerdict).ainvoke(
        _classifier_messages(text)
    )
    return _parse_verdict(result)
