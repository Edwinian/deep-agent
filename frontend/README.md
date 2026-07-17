# Deep Agents frontend

Next.js (App Router) UI for the Deep Agents API. Pages: **Chats**, **Agents**, **System Prompts**, **Skills**.

## Setup

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173 — Next.js rewrites `/chats/*` (and `/speech-to-text/*`) to the FastAPI backend at http://127.0.0.1:8000 (configurable via `BACKEND_API_BASE_URL`).

**SSE note:** `/chats/stream` and `/chats/cancel-stream` call FastAPI directly (`http://127.0.0.1:8000` by default, override with `NEXT_PUBLIC_STREAM_API_BASE_URL`). Next.js rewrites buffer `text/event-stream` and would deliver all chunks at once.

Make sure the API is running (`./start.sh` or `python main.py`).

## Structure

```
src/
  app/
    layout.tsx           # Root layout with sidebar shell
    page.tsx             # Redirects to /chats
    chats/page.tsx       # Chatbot page (wraps ChatClient)
    agents/              # Placeholder
    system-prompts/      # Placeholder
    skills/              # Placeholder
    globals.css          # Global styles (ported from Vite App.css + sidebar styles)
  components/
    Sidebar.tsx          # Client nav using next/navigation
    ChatClient.tsx       # Full chat UI (streaming + HITL + voice)
  api.ts                 # API client (fetch-based SSE streaming)
  types.ts               # Shared types
```

## Chat behavior

- First message creates a client `thread_id` and streams via `POST /chats/stream`
- The `threadId` is synced to the URL query string (`/chats?threadId=…`) for shareable links
- Interrupt chunks show approve / edit / reject / respond controls, then resume `/chats/stream` with `permissions`
- **Stop** aborts the fetch and calls `POST /chats/cancel-stream/{thread_id}`
