"""Resolve DeepAgent specs from SQLite configuration."""

from __future__ import annotations

from db.agent_store import AgentNotFoundError, get_agent, get_skills, get_system_prompt
from agents.types import DeepAgent
from tools.tool_registry import resolve_tools
from tools.web_search_tool import get_today_str
from utils.daytona_sandbox import load_skills


def _format_system_prompt(content: str) -> str:
    if "{date}" in content:
        return content.format(date=get_today_str())
    return content


async def resolve_agent(
    agent_id: int,
    *,
    _visited: set[int] | None = None,
) -> DeepAgent:
    """Load an agent spec from the database and resolve tools and subagents."""
    visited = _visited or set()
    if agent_id in visited:
        raise ValueError(f"Cycle detected while resolving agent_id: {agent_id}")
    visited.add(agent_id)

    row = get_agent(agent_id)
    prompt = get_system_prompt(row.system_prompt_id)
    tools = await resolve_tools(row.tool_ids) if row.tool_ids else []

    spec: DeepAgent = {
        "name": row.name,
        "description": row.description,
        "system_prompt": _format_system_prompt(prompt.content),
        "tools": tools,
    }
    if row.model:
        spec["model"] = row.model

    if row.subagent_ids:
        subagents: list[DeepAgent] = []
        for subagent_id in row.subagent_ids:
            subagents.append(await resolve_agent(subagent_id, _visited=visited))
        spec["subagents"] = subagents

    if row.skill_ids:
        skill_rows = get_skills(row.skill_ids)
        spec["skill_paths"] = load_skills(skill_rows)

    return spec


__all__ = ["AgentNotFoundError", "resolve_agent"]
