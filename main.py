"""FastAPI entrypoint for agent invocation."""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.agent_store import init_agent_db
from modules.routers import routers
from utils.get_checkpointer import close_sqlite_checkpointer, init_sqlite_checkpointer


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize persistent SQLite checkpoints for HITL thread continuity."""
    init_agent_db()
    await init_sqlite_checkpointer()
    yield
    await close_sqlite_checkpointer()


app = FastAPI(title="Deep Agents API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

[app.include_router(router) for router in routers]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
