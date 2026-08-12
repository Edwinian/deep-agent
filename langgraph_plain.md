# Plain LangGraph Dummy Agent (Human-Friendly Overview)

This repo mostly builds “production agents” using the `deepagents` helper (`utils/compile_agent.py`), which hides some of the LangGraph wiring.

To make it easier to understand what a production-quality agent still needs *even with plain LangGraph*, this project includes a **dummy plain LangGraph agent**:

- `graphs/dummy_langgraph_agent.py`
- documented in `langgraph_doc.md` (more technical)

This document explains the same dummy agent in **human language**.

## What problem this dummy agent is meant to solve

If you use plain LangGraph directly, you still need to handle the “production features” that a real product agent requires, such as:

- pausing when a sensitive tool should be approved (HITL)
- not crashing when the planner produces incomplete tool arguments (tool-arg repair)
- keeping consistent state across turns (checkpointing)
- keeping PII from leaking during planning (basic redaction)

The dummy agent is a small, rule-based example that demonstrates those behaviors end-to-end.

## What it can do

The dummy agent can handle one simple kind of decision:

1. If the user message does **not** look “sensitive”, it responds immediately with a normal message.
2. If the user message **does** look sensitive, it “plans” a tool action and then **pauses** for human approval before executing it.

This makes it easy to test and demonstrate the full tool-call → HITL → tool execution loop.

## How human approval (HITL) works here

When the agent decides a sensitive action is needed, it:

1. Builds a “planned tool call”
2. If that tool is configured as requiring approval, the agent **interrupts** execution
3. The running system returns an “awaiting approval” response to the client
4. On resume, the client sends a decision (approve/reject/etc.)
5. The agent continues and the tool is executed only if approved

From your app’s perspective, this behaves like any other production agent that supports tool approval.

## What “tool-call lifecycle” means in this dummy

Even though it’s a dummy, it still models the real life cycle:

- **Planning:** the agent emits “I want to call tool X with these arguments”
- **Repair (before HITL):** if planned tool arguments are missing, it fills in safe defaults *before* HITL checks happen
- **Interrupt (if needed):** HITL pauses execution so a user can approve/reject
- **Execute:** after approval, it runs the tool and appends the tool result to the chat

This is important because in a production system, you want “approval logic” to see consistent, valid tool calls.

## What PII detection/redaction does here

The dummy agent does very light PII handling:

- it redacts email addresses in the user text before making the “sensitive planning” decision

This is only meant as a demo of the idea that “planning can’t leak raw user PII.”

The production system has stronger PII middleware; the dummy agent keeps things simple so it runs everywhere.

## Checkpointing: why interrupts can be resumed

To support “pause and then continue later”, the dummy agent must store enough execution state so it can resume correctly.

That’s what checkpointing is for.

In practice:

- the dummy graph is compiled with a checkpointer
- interrupts can be resumed on later invocations using the same `thread_id`

## Subagents: what the dummy does (and does not) demonstrate

The production system you already have supports subagents (delegation to other compiled agents) and streams them specially.

This dummy focuses on HITL + tool-call lifecycle + state contract compatibility, not subagent streaming.

So if you’re trying to verify subagent-specific streaming events, this dummy may not show the same subagent projections as a `deepagents`-compiled agent.

## How you would invoke/stream it in your app

Your existing streaming and invoke endpoints are designed to work with a **compiled LangGraph graph** object.

That means you can integrate the dummy agent without rewriting your API layer:

- the dummy graph can be passed anywhere the app expects a compiled LangGraph agent
- the app can still stream messages/tool calls and still surface interrupts for human approval

## When to use this dummy agent

Use it when you want to:

- understand the shape of a “plain LangGraph production loop”
- demo HITL tool approval without the full production complexity
- experiment with how state, interrupts, and tool execution interact

