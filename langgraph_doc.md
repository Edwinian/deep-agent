# Dummy Plain LangGraph Agent (Graphs Folder)

This repo primarily uses `utils/compile_agent.py` + `deepagents.create_deep_agent()`.  
For learning and testing, the `graphs/` folder contains a **plain LangGraph** dummy agent that is wired to be compatible with the existing runtime in:

- `modules/chats/invoke_service.py`
- `modules/chats/stream_service.py`

The important part is **not** that it matches deepagents feature-for-feature, but that it follows the same *runtime/event contract* those services expect.

## 1) Files

- `graphs/dummy_langgraph_agent.py`: builds a compiled LangGraph graph
- (This document) `langgraph_doc.md`: describes contract + feature behavior

## 2) State schema conventions

`modules/chats/invoke_service.py` always invokes the graph with:

```json
{ "messages": [HumanMessage(...)] }
```

So the dummy agent uses this state shape:

- `messages`: `Annotated[list[BaseMessage], add_messages]`
- `files` (optional): `dict[str, Any]` (not required by this dummy)
- `todos` (optional): `list[Any]` (not required by this dummy)

HITL interrupts follow LangGraph’s reserved field:

- `__interrupt__` is produced by `langgraph.types.interrupt(...)`

## 3) Subagent handling

This dummy agent does **not** implement deepagents’ “subgraph projection” (`run.subagents`), so `StreamService` will not emit `subagent_*` SSE chunks for this dummy agent.

However, the main agent’s:

- `messages`
- `tool_calls`
- `values` (interrupt surfacing)

projections are compatible, so streaming/invocation still work end-to-end.

If you later need actual `run.subagents` projection compatibility, you must either:

- replicate deepagents’ subgraph/projection wiring, or
- invoke precompiled subgraphs in the same structural way deepagents does.

## 4) Tool-call lifecycle + tool-arg repair relative to HITL

Deepagents uses middleware such as `ToolCallArgsRepairMiddleware` so that the model’s planned `tool_calls` are valid **before** HITL inspects them.

The dummy agent mirrors that ordering inside `_planner_node`:

1. planner emits an `AIMessage` with a `tool_calls` entry
2. `_repair_tool_call_args(...)` fills missing required args **before** HITL checks
3. if an interruptible tool is present, the agent calls:
   - `langgraph.types.interrupt({ "action_requests": [...] })`
4. on resume, the planner applies decisions to the planned tool_calls
5. routing continues into the `ToolNode` for approved tool calls

So: **repair happens before interrupt/HITL**, matching the nuance in `utils/task_tool_args_repair.py`.

Tools passed to `ToolNode` are wrapped with `utils.retry.wrap_tool_with_retry`, the same helper used for MCP/hotel tools in the deepagents path (`tools/tool_registry.py`). Transient network failures get exponential backoff; permanent errors still fail fast.
## 5) Interrupt payloads (what `run.interrupts()` / `run.values` look like)

`StreamService` and `InvokeService` treat interrupts as either:

- `langgraph.types.Interrupt` objects, or
- dict payloads with `{ "id": ..., "value": ... }`

They then extract:

- `action_requests` from the interrupt payload value

The dummy agent uses the standard LangGraph API:

- `resume_payload = interrupt({"action_requests": action_requests})`

This produces LangGraph interrupts that `modules/chats/stream_service.py` can surface via:

- `StreamService.extract_interrupts(...)`
- `utils.hitl.collect_action_requests(...)`

Resume:

When the client resumes, `modules/chats/invoke_service.py` builds:

```python
Command(resume={interrupt_id: {"decisions": decisions}})
```

LangGraph then re-executes the node, and the corresponding `interrupt(...)` call returns that object as `resume_payload`.

The dummy agent applies decisions by order.

## 6) Checkpointing

To make `interrupt(...)` resumable, LangGraph requires compiling the graph with a checkpointer.

In `graphs/dummy_langgraph_agent.py`, `build_dummy_langgraph_agent(checkpointer=...)`
compiles with:

- `builder.compile(checkpointer=checkpointer)`

so interrupts can pause/resume across turns.

## 7) How to compile the dummy agent

Example usage (outside the FastAPI services):

```python
from graphs.dummy_langgraph_agent import build_dummy_langgraph_agent
from utils.get_checkpointer import get_checkpointer, CheckpointerType

checkpointer = get_checkpointer(CheckpointerType.ASYNC_SQLITE)
graph = build_dummy_langgraph_agent(checkpointer=checkpointer)
```

Then you can invoke/stream it with `agent.ainvoke(...)` / `agent.astream_events(...)` using the same `StreamService` logic.

