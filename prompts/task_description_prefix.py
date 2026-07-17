"""Tool description for the deepagents `task` delegation tool.

Keep this short: long built-in deepagents copy causes some models (incl. Grok)
to omit required ``description`` / ``subagent_type`` args.
"""

TASK_TOOL_DESCRIPTION = """\
Delegate a task to an isolated subagent. You MUST pass both required arguments:
- description: detailed standalone instructions for the subagent (string)
- subagent_type: one of the agent names listed below (string)

Examples:
  task(
    description="What does Lilian Weng say about types of reward hacking? Answer by looking up on vector DB.",
    subagent_type="rag_agent"
  )
  task(
    description="Find which teams reached the FIFA World Cup 2026 final. Return team names only.",
    subagent_type="research_agent"
  )

Never call task with empty arguments.

Available agents:
{available_agents}
"""

# Backward-compatible alias used by older imports / docs.
TASK_DESCRIPTION_PREFIX = TASK_TOOL_DESCRIPTION
