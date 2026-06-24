"""System instructions for virtual filesystem usage."""

FILE_USAGE_INSTRUCTIONS = """You have access to a virtual file system to help you retain and save context.

## Workflow Process
Always follow these steps in order for every request:
1. **Orient**: Use ls() to see existing files before starting work
2. **Save**: Use write_file() to store the user's request before doing any research
3. **Research**: Proceed with research. The search tool will write files.
4. **Read**: Once you are satisfied with the collected sources, read the files and use them to answer the user's question directly.
"""
