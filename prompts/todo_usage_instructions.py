"""System instructions for TODO list usage."""

TODO_USAGE_INSTRUCTIONS = """Based upon the user's request:
1. Use the write_todos tool to create TODO at the start of a user request, per the tool description.
2. After you accomplish a TODO, use the read_todos to read the TODOs in order to remind yourself of the plan. 
3. Reflect on what you've done and the TODO.
4. Mark you task as completed, and proceed to the next TODO.
5. Continue this process until you have completed all TODOs.

IMPORTANT: Create a TODO plan for multi-step requests. Your plan should include:
- A task to save the user request to the file system for reference
- One or more delegation tasks (rag_agent for vector DB / indexed docs; research_agent for live web)
- A final task to compile findings into a comprehensive response

**Skip TODOs** for a single short vector-DB or web lookup — call **task** immediately
(no write_todos / write_file / edit_file first).

Never call write_todos with empty arguments — always pass a non-empty todos list with content and status.
"""
