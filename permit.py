"""Test client for resuming interrupted tool calls via POST /invoke."""

from __future__ import annotations

import json
import sys

import httpx

from agents.general_agent import GENERAL_AGENT_ID
from schemas.invoke_request import DecisionType, Permission

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_THREAD_ID = "915491e2-2060-41da-8e3e-6e1e629993ec"
DEFAULT_PERMISSIONS: list[Permission] = [
    {
        "name": "web_search_tool",
        "decision": DecisionType.APPROVE,
    }
]


def permit_agent(
    *,
    thread_id: str,
    permissions: list[Permission],
    agent_id: int = GENERAL_AGENT_ID,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 300.0,
) -> dict:
    """Resume an interrupted thread by approving, editing, rejecting, or responding to tool calls."""
    payload = {
        "agent_id": agent_id,
        "thread_id": thread_id,
        "permissions": permissions,
    }

    response = httpx.post(
        f"{base_url.rstrip('/')}/invoke",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    thread_id = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_THREAD_ID
    permissions = DEFAULT_PERMISSIONS

    if not thread_id:
        print("thread_id is required. Set DEFAULT_THREAD_ID or pass it as argv[2].", file=sys.stderr)
        return 1
    if not permissions:
        print("permissions is required. Set DEFAULT_PERMISSIONS in permit.py.", file=sys.stderr)
        return 1

    print(f"POST {base_url}/invoke")
    print(f"agent_id={GENERAL_AGENT_ID}")
    print(f"thread_id={thread_id!r}")
    print(f"permissions={permissions!r}")
    print()

    try:
        result = permit_agent(
            base_url=base_url,
            thread_id=thread_id,
            permissions=permissions,
        )
    except httpx.HTTPStatusError as exc:
        print(f"HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1
    except httpx.RequestError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        print("Is the API running? Try: python main.py", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
