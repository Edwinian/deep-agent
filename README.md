# Deep Agents

Production LangGraph deep-agent framework with task planning, virtual file context offloading, and sub-agent delegation.

## Quickstart

### Prerequisites

- Python 3.11 or later

```bash
python3 --version
```

### Installation

1. Clone the repository and enter the project directory.

2. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:

```env
TAVILY_API_KEY=your_tavily_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
XAI_API_KEY=your_xai_api_key_here

# Optional
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGCHAIN_TRACING_V2=true
LANGSMITH_PROJECT=deep-agents-from-scratch
```

### Run API (dev, auto-reload)

```bash
python main.py
```

Or:

```bash
uvicorn main:app --reload
```


## Project Layout

```
agents/       Agent specs and registry
tools/        Agent tools (search, todos, reflection)
prompts/      System prompts and tool descriptions
utils/        Compilation, display, and checkpointer helpers
state.py      Shared agent state schema
requirements.txt
```

## Patterns

- **Task planning** — TODO lists with status tracking
- **Context offloading** — Virtual file system in agent state
- **Context isolation** — Sub-agent delegation for focused workflows
