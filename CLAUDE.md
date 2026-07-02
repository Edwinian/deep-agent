# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production LangGraph deep-agent framework. Agent specs, tools, prompts, and compilation utilities live at the project root.

## Development Commands

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Code Quality
```bash
# If ruff/mypy are installed in your environment
ruff check .
ruff format .
mypy .
```

## Architecture

### Core Components

**State Management (`state.py`)**
- `DeepAgentState`: Extends LangGraph's `AgentState` with todos and files
- `Todo`: TypedDict for task tracking with status (pending/in_progress/completed)
- `file_reducer`: Merges file dictionaries in state updates

**Agents (`agents/`)**
- `research_agent.py`: Research sub-agent with web search and reflection
- `general_agent.py`: Orchestrator with todos, files, and sub-agent delegation
- `agent_registry.py`: Registry of agent specs keyed by ID

**Tools (`tools/`)**
- `web_search_tool.py`: Tavily search with context offloading to virtual files
- `think_tool.py`: Strategic reflection for research workflows
- `todo_tools.py`: Read TODO list from agent state
- `default_interrupt_on.py`: Human-in-the-loop interrupt defaults

**Utils (`utils/`)**
- `compile_agent.py`: Compile a `DeepAgent` spec into a runnable graph
- `compile_subagents.py`: Recursively compile nested subagents
- `display.py`: Rich formatting for messages and prompts
- `get_checkpointer.py`: LangGraph checkpointer factory

### Key Patterns

- Context offloading to virtual files stored in state
- TODO lists for planning and progress tracking
- Subagent spawning for context quarantine
- Task-specific prompt engineering

## Environment Variables

Create `.env` in the project root:
```bash
TAVILY_API_KEY=your_tavily_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
XAI_API_KEY=your_xai_api_key_here

# Optional
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=deep-agents-from-scratch
```

## Important Notes

- Virtual file system is ephemeral — exists only during agent execution
- When adding a new agent to `agents/`, register it in `agent_registry.py`
- When adding a new tool to `tools/`, add it to `DEFAULT_INTERRUPT_ON` in `default_interrupt_on.py`
- Import statements always at the top of the file
