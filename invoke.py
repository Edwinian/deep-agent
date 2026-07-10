"""Test client for POST /invoke against the general agent."""

from __future__ import annotations

import json
import sys

import httpx

from agents.ids import GENERAL_AGENT_ID

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MESSAGE = (
    "Search hotels for me from database located in Zurich"
)


def invoke_agent(
    *,
    base_url: str = DEFAULT_BASE_URL,
    agent_id: int = GENERAL_AGENT_ID,
    message: str = DEFAULT_MESSAGE,
    thread_id: str | None = None,
    model_config: dict | None = None,
    timeout: float = 300.0,
) -> dict:
    """Call the /invoke endpoint and return the JSON response.

    Reuse ``thread_id`` across calls for the same user session. Omit it only for
    a one-off request; the server will assign one. Call DELETE /threads/{id}
    when the session ends.
    """
    payload: dict = {
        "agent_id": agent_id,
        "message": message,
    }
    if model_config is not None:
        payload["model_config"] = model_config

    response = httpx.post(
        f"{base_url.rstrip('/')}/invoke",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    message = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MESSAGE

    print(f"POST {base_url}/invoke")
    print(f"agent_id={GENERAL_AGENT_ID}")
    print(f"message={message!r}")
    print()

    try:
        result = invoke_agent(base_url=base_url, message=message)
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
