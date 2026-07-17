"""System instructions for virtual filesystem usage."""

FILE_USAGE_INSTRUCTIONS = """You have access to a virtual file system to help you retain and save context.

Paths are virtual (e.g. ``/user_request.txt``). When Daytona sandboxes are enabled,
the backend maps them to a per-thread workspace inside the sandbox automatically.

## Workflow Process

**Short-circuit for simple requests** — when the user asks one short, direct question
(under ~15 words, no comparison or multi-part analysis):
- If they mention vector DB / Chroma / indexed documents → **task** with `subagent_type="rag_agent"`.
- If they need live web facts (e.g. "latest football news") → **task** with `subagent_type="research_agent"`.
- **Skip** ls, but **still Save**: use write_file(file_path="/user_request.txt", content=<user request>) before delegating.

For all other requests, follow these steps in order:
1. **Orient**: Use ls(path="/") to see existing files before starting work
2. **Save**: Use write_file(file_path="/user_request.txt", content=<user request>) before delegating
3. **Delegate**: Call **task** with the correct subagent (`rag_agent` for vector DB; `research_agent` for live web).
4. **Read**: Once the subagent returns, read any saved files and answer the user's question directly.

Never call ls, write_file, or write_todos with empty arguments — always include the required parameters.
"""
