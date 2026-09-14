"""Moderation middleware — flag hate speech, harassment, and violence."""

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

MODERATION_REFUSAL = (
    "I cannot process that request because it contains harmful or inappropriate "
    "content."
)

CLASSIFIER_SYSTEM_PROMPT = """\
You are a moderation classifier for an AI agent.

Flag harmful or inappropriate inputs (hate speech, harassment, violence) to \
maintain safe, respectful interactions.

Mark flagged=true when the user input:
- Contains or solicits hate speech (attacks on people based on protected \
characteristics)
- Harasses, threatens, or bullies a person or group
- Solicits, instructs, or celebrates violence or violent crime
- Sexual content involving minors, or self-harm instructions

Mark flagged=false for ordinary questions, including factual or news research \
that mentions violence or conflict without soliciting harm (for example \
"What is the death toll in yesterday's earthquake?").

Set category to one of: hate, harassment, violence, sexual, self_harm, none.
Return only the structured verdict. Do not answer the user's request.
"""

_DEFAULT_CLASSIFIER_MODEL = ModelName.GROK_4_FAST_NON_REASONING.with_provider()


class ModerationVerdict(BaseModel):
    """Structured classifier output."""

    flagged: bool = Field(
        description=(
            "True if the input is harmful or inappropriate "
            "(hate speech, harassment, violence, or similar)."
        )
    )
    category: str = Field(
        default="none",
        description="hate, harassment, violence, sexual, self_harm, or none.",
    )
    reason: str = Field(
        default="",
        description="Short explanation of why the input is flagged or allowed.",
    )


class ModerationMiddleware(AgentMiddleware):
    """Block harmful user inputs before the agent runs.

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
            verdict = classify_moderation(text, model=self._classifier())
        except Exception:
            logger.exception("Moderation classifier failed; allowing the request")
            return None
        if verdict.flagged:
            logger.warning(
                "Blocked moderated input category=%s: %s",
                verdict.category,
                verdict.reason,
            )
            return jump_to_end(MODERATION_REFUSAL)
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
            verdict = await aclassify_moderation(text, model=self._classifier())
        except Exception:
            logger.exception("Moderation classifier failed; allowing the request")
            return None
        if verdict.flagged:
            logger.warning(
                "Blocked moderated input category=%s: %s",
                verdict.category,
                verdict.reason,
            )
            return jump_to_end(MODERATION_REFUSAL)
        return None


def _classifier_messages(text: str) -> list[Any]:
    return [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=text),
    ]


def _parse_verdict(result: Any) -> ModerationVerdict:
    if isinstance(result, ModerationVerdict):
        return result
    if isinstance(result, dict):
        return ModerationVerdict.model_validate(result)
    return ModerationVerdict.model_validate(
        {
            "flagged": False,
            "category": "none",
            "reason": "Classifier returned an unexpected payload.",
        }
    )


@observe(name="classify_moderation")
@traceable(name="classify_moderation")
def classify_moderation(text: str, *, model: Any) -> ModerationVerdict:
    """Return a structured moderation verdict for ``text``."""
    result = model.with_structured_output(ModerationVerdict).invoke(
        _classifier_messages(text)
    )
    return _parse_verdict(result)


@observe(name="classify_moderation")
@traceable(name="classify_moderation")
async def aclassify_moderation(text: str, *, model: Any) -> ModerationVerdict:
    """Async structured moderation verdict for ``text``."""
    result = await model.with_structured_output(ModerationVerdict).ainvoke(
        _classifier_messages(text)
    )
    return _parse_verdict(result)
