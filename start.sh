#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python"
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.env"
  set +a
fi

WEATHER_MCP_URL="${WEATHER_MCP_URL:-http://127.0.0.1:8001/mcp}"
if [[ "$WEATHER_MCP_URL" =~ :([0-9]+) ]]; then
  WEATHER_PORT="${BASH_REMATCH[1]}"
else
  WEATHER_PORT=8001
fi

free_port() {
  local port=$1
  local pids
  pids=$(lsof -ti ":$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "Stopping process(es) on port $port..."
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

cleanup() {
  if [[ -n "${WEATHER_PID:-}" ]] && kill -0 "$WEATHER_PID" 2>/dev/null; then
    kill "$WEATHER_PID" 2>/dev/null || true
    wait "$WEATHER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

free_port "$WEATHER_PORT"

"$PYTHON" mcp_servers/weather_server.py &
WEATHER_PID=$!

for _ in {1..30}; do
  if kill -0 "$WEATHER_PID" 2>/dev/null && lsof -ti ":$WEATHER_PORT" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$WEATHER_PID" 2>/dev/null; then
    echo "Weather MCP server failed to start on port $WEATHER_PORT" >&2
    exit 1
  fi
  sleep 0.2
done

echo "Weather MCP server running at $WEATHER_MCP_URL"

"$PYTHON" main.py
