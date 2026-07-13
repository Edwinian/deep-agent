"""Seed default agent rows into SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from agents.ids import GENERAL_AGENT_ID, RAG_AGENT_ID, RESEARCH_AGENT_ID
from constants.model_name import ModelName
from prompts import (
    FILE_USAGE_INSTRUCTIONS,
    RAG_AGENT_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
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


def _build_general_system_prompt() -> str:
    subagent_instructions = SUBAGENT_USAGE_INSTRUCTIONS.format(
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
        max_researcher_iterations=MAX_RESEARCHER_ITERATIONS,
        date=datetime.now().strftime("%a %b %-d, %Y"),
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
    )


def seed_default_agents(conn: sqlite3.Connection) -> None:
    """Insert default research and general agent configuration."""
    research_prompt = RESEARCHER_INSTRUCTIONS
    general_prompt = _build_general_system_prompt()
    rag_prompt = RAG_AGENT_INSTRUCTIONS.format(
        date=datetime.now().strftime("%a %b %-d, %Y"),
    )
    now = datetime.now(timezone.utc).isoformat()

    conn.executemany(
        "INSERT INTO SystemPrompt (id, name, content, created_at) VALUES (?, ?, ?, ?)",
        [
            (RESEARCH_SYSTEM_PROMPT_ID, "research-agent", research_prompt, now),
            (GENERAL_SYSTEM_PROMPT_ID, "general-agent", general_prompt, now),
            (RAG_SYSTEM_PROMPT_ID, "rag-agent", rag_prompt, now),
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
                "research-agent",
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
                "rag-agent",
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
                "general-agent",
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
