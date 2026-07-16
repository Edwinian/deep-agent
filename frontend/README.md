# Deep Agents frontend

React chat UI for the Deep Agents API (`POST /stream`, HITL resume, `POST /cancel-stream/{thread_id}`).

## Setup

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173 — Vite proxies API calls to http://127.0.0.1:8000.

Make sure the API is running (`./start.sh` or `python main.py`).

## Behavior

- First message creates a client `thread_id` and streams via `POST /stream`
- Later messages reuse that `thread_id`
- Interrupt chunks show approve / edit / reject / respond controls, then resume `/stream` with `permissions`
- **Stop** aborts the fetch and calls `POST /cancel-stream/{thread_id}`
