"""General tool-output quality evaluation with query rewrite and bounded retry.

Unlike infrastructure backoff (``utils.retry``), this loop retries when the
*semantic* quality of a tool's text output is too low relative to the user query.

Use ``run_with_quality_retry`` inside tools, or ``wrap_tool_with_quality_retry``
for LangChain ``BaseTool`` instances that take a primary string argument
(``query``, ``question``, ``city``, etc.).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from constants.model_name import ModelName
from utils.retry import is_transient_exception
from utils.tracing import trace

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = ModelName.GROK_3_MINI.with_provider()

DEFAULT_MAX_QUALITY_RETRIES = int(os.getenv("TOOL_QUALITY_MAX_RETRIES", "1"))
DEFAULT_MIN_QUALITY_SCORE = float(os.getenv("TOOL_QUALITY_MIN_SCORE", "0.6"))

DEFAULT_QUERY_ARG_KEYS = ("query", "question", "q", "search", "input", "city", "location")

OnStatus = Callable[[str], None]
ExecuteFn = Callable[[str], str]


@dataclass
class ToolOutputEvaluation:
    """Result of scoring a tool's text output against the user query."""

    score: float
    feedback: str
    is_acceptable: bool


@dataclass
class QualityRetryResult:
    """Outcome of ``run_with_quality_retry``."""

    output: str
    query_used: str
    original_query: str
    attempts: int
    scores: list[float] = field(default_factory=list)
    feedback: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(str(self.output or "").strip())


class _EvaluationModel(BaseModel):
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality / relevance score from 0.0 (useless) to 1.0 (excellent).",
    )
    feedback: str = Field(
        description="Brief reason for the score and how to improve the query if low.",
    )


class _RewriteModel(BaseModel):
    rewritten_query: str = Field(
        description="Improved query that should yield better tool results.",
    )


def _get_eval_model():
    from utils.resolve_model import resolve_model

    model = resolve_model(model=_DEFAULT_MODEL)
    if model is None:
        raise RuntimeError("No model configured for tool output evaluation")
    return model


@trace("evaluate_tool_output")
def evaluate_tool_output(
    user_query: str,
    tool_output: str,
    *,
    tool_name: str = "tool",
    min_score: float = DEFAULT_MIN_QUALITY_SCORE,
) -> ToolOutputEvaluation:
    """Score whether ``tool_output`` adequately answers / addresses ``user_query``."""
    text = str(tool_output or "").strip()
    if not text:
        return ToolOutputEvaluation(
            score=0.0,
            feedback="Tool returned empty output.",
            is_acceptable=False,
        )

    prompt = (
        f"You evaluate tool output quality for the tool `{tool_name}`.\n"
        f"User query:\n{user_query}\n\n"
        f"Tool output:\n{text[:6000]}\n\n"
        "Score how well the output addresses the user query "
        "(relevance, usefulness, specificity). "
        "If the output is an error message or clearly off-topic, score low."
        " Always return both `score` (0.0-1.0) and `feedback`."
    )
    model = _get_eval_model()
    try:
        result = model.with_structured_output(_EvaluationModel).invoke(
            [{"role": "user", "content": prompt}]
        )
        if isinstance(result, dict):
            result = _EvaluationModel.model_validate(
                {
                    "score": result.get("score", 0.5),
                    "feedback": result.get("feedback") or "No feedback.",
                }
            )
        score = float(result.score)
        feedback = str(result.feedback or "").strip() or "No feedback."
    except Exception as exc:
        # Some providers return empty structured payloads; don't fail the tool.
        logger.warning(
            "evaluate_tool_output structured parse failed for %s: %s",
            tool_name,
            exc,
        )
        score = 0.65 if len(text) > 40 else 0.35
        feedback = f"Evaluator unavailable ({type(exc).__name__}); used length heuristic."

    return ToolOutputEvaluation(
        score=score,
        feedback=feedback,
        is_acceptable=score >= min_score,
    )


@trace("rewrite_tool_query")
def rewrite_tool_query(
    user_query: str,
    *,
    tool_name: str = "tool",
    tool_output: str = "",
    feedback: str = "",
) -> str:
    """Rewrite ``user_query`` to improve the next tool attempt."""
    prompt = (
        f"Rewrite the user query for another call to tool `{tool_name}`.\n"
        f"Original query:\n{user_query}\n\n"
        f"Previous tool output (may be empty or low quality):\n{str(tool_output)[:3000]}\n\n"
        f"Evaluator feedback:\n{feedback}\n\n"
        "Return one improved query string only. Keep it concise and searchable."
        " Always populate `rewritten_query`."
    )
    model = _get_eval_model()
    try:
        result = model.with_structured_output(_RewriteModel).invoke(
            [{"role": "user", "content": prompt}]
        )
        if isinstance(result, dict):
            result = _RewriteModel.model_validate(
                {"rewritten_query": result.get("rewritten_query") or user_query}
            )
        rewritten = str(result.rewritten_query or "").strip()
    except Exception as exc:
        logger.warning(
            "rewrite_tool_query structured parse failed for %s: %s",
            tool_name,
            exc,
        )
        rewritten = ""
    if not rewritten:
        # Mild fallback rewrite so retries still change something.
        rewritten = f"{user_query.strip()} key facts passages"
    return rewritten


def run_with_quality_retry(
    query: str,
    execute: ExecuteFn,
    *,
    tool_name: str = "tool",
    max_retries: int = DEFAULT_MAX_QUALITY_RETRIES,
    min_score: float = DEFAULT_MIN_QUALITY_SCORE,
    on_status: OnStatus | None = None,
) -> QualityRetryResult:
    """Execute ``execute(query)``, evaluate output, rewrite + retry while score is low.

    ``max_retries`` is the number of *rewrites* allowed after the first attempt
    (total attempts = ``max_retries + 1``), matching RAG ``MAX_QUERY_REWRITES``.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    original_query = query
    current_query = query
    scores: list[float] = []
    last_feedback: str | None = None
    last_output = ""
    attempts = 0

    while True:
        attempts += 1
        if on_status is not None:
            on_status(f"Running {tool_name} (attempt {attempts})…")

        try:
            last_output = str(execute(current_query) or "")
        except Exception as exc:
            # Permanent failures should not burn quality rewrite budget.
            if not is_transient_exception(exc):
                raise
            logger.exception("%s failed during quality-retry attempt %s", tool_name, attempts)
            last_output = ""
            last_feedback = f"{type(exc).__name__}: {exc}"
            scores.append(0.0)
            rewrites_left = max_retries - (attempts - 1)
            if rewrites_left <= 0:
                return QualityRetryResult(
                    output="",
                    query_used=current_query,
                    original_query=original_query,
                    attempts=attempts,
                    scores=scores,
                    feedback=last_feedback,
                    error=f"{tool_name} failed after {attempts} attempt(s): {last_feedback}",
                )
            if on_status is not None:
                on_status(f"{tool_name} failed, rewriting query and retrying…")
            current_query = rewrite_tool_query(
                original_query,
                tool_name=tool_name,
                tool_output=last_feedback,
                feedback=last_feedback,
            )
            continue

        if on_status is not None:
            on_status(f"Evaluating {tool_name} output quality…")

        try:
            evaluation = evaluate_tool_output(
                original_query,
                last_output,
                tool_name=tool_name,
                min_score=min_score,
            )
        except Exception as exc:
            logger.warning(
                "evaluate_tool_output raised for %s; accepting output: %s",
                tool_name,
                exc,
            )
            evaluation = ToolOutputEvaluation(
                score=1.0 if str(last_output or "").strip() else 0.0,
                feedback=f"Evaluation failed ({type(exc).__name__}); accepted raw output.",
                is_acceptable=bool(str(last_output or "").strip()),
            )
        scores.append(evaluation.score)
        last_feedback = evaluation.feedback

        if evaluation.is_acceptable:
            return QualityRetryResult(
                output=last_output,
                query_used=current_query,
                original_query=original_query,
                attempts=attempts,
                scores=scores,
                feedback=evaluation.feedback,
            )

        rewrites_left = max_retries - (attempts - 1)
        if rewrites_left <= 0:
            return QualityRetryResult(
                output=last_output,
                query_used=current_query,
                original_query=original_query,
                attempts=attempts,
                scores=scores,
                feedback=evaluation.feedback,
                error=(
                    f"{tool_name} output quality too low after {attempts} attempt(s) "
                    f"(scores={scores}; min={min_score}). {evaluation.feedback}"
                ),
            )

        if on_status is not None:
            on_status(
                f"{tool_name} quality low ({evaluation.score:.2f}), "
                "rewriting query and retrying…"
            )
        current_query = rewrite_tool_query(
            original_query,
            tool_name=tool_name,
            tool_output=last_output,
            feedback=evaluation.feedback,
        )


def _tool_input_as_dict(tool_input: object) -> dict[str, Any]:
    if isinstance(tool_input, dict):
        return dict(tool_input)
    if isinstance(tool_input, str):
        return {"query": tool_input}
    return {}


def _find_query_key(args: dict[str, Any], query_keys: tuple[str, ...]) -> str | None:
    for key in query_keys:
        if key in args and isinstance(args[key], str):
            return key
    return None


def _extract_tool_result_text(result: object) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, ToolMessage):
        content = result.content
        return content if isinstance(content, str) else str(content)
    update = getattr(result, "update", None)
    if isinstance(update, dict):
        messages = update.get("messages") or []
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, ToolMessage):
                    content = message.content
                    return content if isinstance(content, str) else str(content)
    if isinstance(result, dict):
        for key in ("content", "output", "result", "message"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return str(result)


def wrap_tool_with_quality_retry(
    tool: BaseTool,
    *,
    query_keys: tuple[str, ...] = DEFAULT_QUERY_ARG_KEYS,
    max_retries: int = DEFAULT_MAX_QUALITY_RETRIES,
    min_score: float = DEFAULT_MIN_QUALITY_SCORE,
    on_status: OnStatus | None = None,
) -> BaseTool:
    """Wrap ``invoke`` / ``ainvoke`` with evaluate → rewrite → retry.

    If none of ``query_keys`` are present in the tool args, the tool runs once
    with no quality loop (safe for tools like ``book-hotel``).
    """
    original_invoke = tool.invoke
    original_ainvoke = tool.ainvoke
    tool_name = str(getattr(tool, "name", None) or "tool")

    def invoke(tool_input: object, config: object = None, **kwargs: object) -> object:
        args = _tool_input_as_dict(tool_input)
        query_key = _find_query_key(args, query_keys)
        if query_key is None:
            return original_invoke(tool_input, config=config, **kwargs)

        final_result: dict[str, object] = {"value": None}

        def execute(query: str) -> str:
            call_args = dict(args)
            call_args[query_key] = query
            result = original_invoke(call_args, config=config, **kwargs)
            final_result["value"] = result
            return _extract_tool_result_text(result)

        quality = run_with_quality_retry(
            str(args[query_key]),
            execute,
            tool_name=tool_name,
            max_retries=max_retries,
            min_score=min_score,
            on_status=on_status,
        )
        if quality.error and not str(quality.output or "").strip():
            raise RuntimeError(quality.error)
        return final_result["value"]

    async def ainvoke(tool_input: object, config: object = None, **kwargs: object) -> object:
        args = _tool_input_as_dict(tool_input)
        query_key = _find_query_key(args, query_keys)
        if query_key is None:
            return await original_ainvoke(tool_input, config=config, **kwargs)

        final_result: dict[str, object] = {"value": None}

        def execute(query: str) -> str:
            # Sync bridge: quality loop is sync; call sync invoke path.
            call_args = dict(args)
            call_args[query_key] = query
            result = original_invoke(call_args, config=config, **kwargs)
            final_result["value"] = result
            return _extract_tool_result_text(result)

        quality = run_with_quality_retry(
            str(args[query_key]),
            execute,
            tool_name=tool_name,
            max_retries=max_retries,
            min_score=min_score,
            on_status=on_status,
        )
        if quality.error and not str(quality.output or "").strip():
            raise RuntimeError(quality.error)
        return final_result["value"]

    # StructuredTool is a Pydantic model; plain setattr rejects non-field names.
    object.__setattr__(tool, "invoke", invoke)
    object.__setattr__(tool, "ainvoke", ainvoke)
    return tool
