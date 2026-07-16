"""Seed default agent rows into SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from agents.ids import GENERAL_AGENT_ID, RAG_AGENT_ID, RESEARCH_AGENT_ID
from constants.agent_name import AgentName
from constants.function_name import FunctionName
from constants.model_name import ModelName
from prompts import (
    FILE_USAGE_INSTRUCTIONS,
    GENERATE_ANSWER,
    GRADE_DOCUMENTS,
    PII_GUARDRAILS,
    RAG_AGENT_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    REWRITE_QUERY,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)
from tools.hotel.get_hotel_tools import HOTEL_TOOLS_ID
from tools.math.get_math_mcp_tools import MATH_MCP_TOOLS_ID
from tools.rag.retrieve_tool import RETRIEVE_TOOL_ID
from tools.think.think_tool import THINK_TOOL_ID
from tools.weather.get_weather_mcp_tools import WEATHER_MCP_TOOLS_ID
from tools.web_search.web_search_tool import WEB_SEARCH_TOOL_ID

DEFAULT_MODEL = ModelName.GROK_4_3.with_provider()
MAX_CONCURRENT_RESEARCH_UNITS = 3
MAX_RESEARCHER_ITERATIONS = 3

RESEARCH_SYSTEM_PROMPT_ID = 1
GENERAL_SYSTEM_PROMPT_ID = 2
RAG_SYSTEM_PROMPT_ID = 3
GENERATE_ANSWER_SYSTEM_PROMPT_ID = 4
GRADE_DOCUMENTS_SYSTEM_PROMPT_ID = 5
REWRITE_QUERY_SYSTEM_PROMPT_ID = 6


def _build_general_system_prompt() -> str:
    """Build the general-agent prompt.

    Leaves a ``{date}`` placeholder for runtime formatting in ``resolve_agent``.
    """
    subagent_instructions = SUBAGENT_USAGE_INSTRUCTIONS.format(
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
        max_researcher_iterations=MAX_RESEARCHER_ITERATIONS,
        date="{date}",
    )
    return (
        "# TODO MANAGEMENT\n"
        + TODO_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# FILE SYSTEM USAGE\n"
        + FILE_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# SUB-AGENT DELEGATION\n"
        + subagent_instructions
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + PII_GUARDRAILS
    )


def _build_research_system_prompt() -> str:
    """Research-agent prompt with runtime ``{date}`` and privacy guardrails."""
    return RESEARCHER_INSTRUCTIONS.rstrip() + "\n\n" + PII_GUARDRAILS


def _build_rag_system_prompt() -> str:
    """RAG-agent prompt with runtime ``{date}`` and privacy guardrails."""
    return RAG_AGENT_INSTRUCTIONS.rstrip() + "\n\n" + PII_GUARDRAILS


def seed_default_agents(conn: sqlite3.Connection) -> None:
    """Insert default research and general agent configuration."""
    research_prompt = _build_research_system_prompt()
    general_prompt = _build_general_system_prompt()
    rag_prompt = _build_rag_system_prompt()
    now = datetime.now(timezone.utc).isoformat()

    conn.executemany(
        "INSERT INTO SystemPrompt (id, name, content, created_at) VALUES (?, ?, ?, ?)",
        [
            (RESEARCH_SYSTEM_PROMPT_ID, AgentName.RESEARCH_AGENT, research_prompt, now),
            (GENERAL_SYSTEM_PROMPT_ID, AgentName.GENERAL_AGENT, general_prompt, now),
            (RAG_SYSTEM_PROMPT_ID, AgentName.RAG_AGENT, rag_prompt, now),
            (GENERATE_ANSWER_SYSTEM_PROMPT_ID, FunctionName.GENERATE_ANSWER, GENERATE_ANSWER, now),
            (GRADE_DOCUMENTS_SYSTEM_PROMPT_ID, FunctionName.GRADE_DOCUMENTS, GRADE_DOCUMENTS, now),
            (REWRITE_QUERY_SYSTEM_PROMPT_ID, FunctionName.REWRITE_QUERY, REWRITE_QUERY, now),
        ],
    )

    conn.executemany(
        """
        INSERT INTO Agent (
            id,
            name,
            description,
            system_prompt_id,
            subagent_ids,
            model,
            tool_ids,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                RESEARCH_AGENT_ID,
                AgentName.RESEARCH_AGENT,
                (
                    "Delegate research to the sub-agent researcher. "
                    "Only give this researcher one topic at a time."
                ),
                RESEARCH_SYSTEM_PROMPT_ID,
                None,
                DEFAULT_MODEL,
                json.dumps([WEB_SEARCH_TOOL_ID, THINK_TOOL_ID]),
                now,
            ),
            (
                RAG_AGENT_ID,
                AgentName.RAG_AGENT,
                (
                    "Delegate questions that need grounded answers from indexed "
                    "documents in the ChromaDB vector store. Use when the user asks "
                    "about content that may already be indexed rather than live web data."
                ),
                RAG_SYSTEM_PROMPT_ID,
                None,
                DEFAULT_MODEL,
                json.dumps([RETRIEVE_TOOL_ID]),
                now,
            ),
            (
                GENERAL_AGENT_ID,
                AgentName.GENERAL_AGENT,
                (
                    "Orchestrates research by managing todos, files, "
                    "and delegating to specialized agents."
                ),
                GENERAL_SYSTEM_PROMPT_ID,
                json.dumps([RESEARCH_AGENT_ID, RAG_AGENT_ID]),
                DEFAULT_MODEL,
                json.dumps([WEATHER_MCP_TOOLS_ID, MATH_MCP_TOOLS_ID, HOTEL_TOOLS_ID]),
                now,
            ),
        ],
    )
