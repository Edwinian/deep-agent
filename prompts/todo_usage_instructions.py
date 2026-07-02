"""System instructions for TODO list usage."""

TODO_USAGE_INSTRUCTIONS = """Based upon the user's request:
1. Use the write_todos tool to create TODO at the start of a user request, per the tool description.
2. After you accomplish a TODO, use the read_todos to read the TODOs in order to remind yourself of the plan. 
3. Reflect on what you've done and the TODO.
4. Mark you task as completed, and proceed to the next TODO.
5. Continue this process until you have completed all TODOs.

IMPORTANT: Always create a research plan of TODOs for any user request. Your plan should always include:
- A task to save the user request to the file system for reference
- One or more research tasks (batch closely related searches into a single TODO)
- A final task to compile findings into a comprehensive response
"""
