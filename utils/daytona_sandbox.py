"""Daytona sandbox provisioning and virtual-to-physical path mapping.

deepagents exposes a virtual filesystem rooted at ``/`` (e.g. ``/user_request.txt``).
Daytona sandboxes run as an unprivileged user and cannot write to the container's
filesystem root. Production setups therefore:

1. Provision a per-thread sandbox with a known writable workspace (option 3).
2. Remap agent virtual paths to that workspace before any file I/O (option 1).

The agent continues to see ``/...`` paths; only the backend translates them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from typing import cast

from daytona import Daytona, DaytonaConfig, Sandbox
from daytona.common.daytona import CreateSandboxFromSnapshotParams
from daytona.common.errors import DaytonaNotFoundError
from daytona.common.sandbox import SandboxState
from deepagents.backends import StateBackend
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from langchain_daytona import DaytonaSandbox
from langgraph.config import get_config

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_AUTO_STOP_INTERVAL_SECONDS = 3600
DEFAULT_WORKSPACE_THREADS_SUBDIR = "threads"

_daytona_client_instance: Daytona | None = None
_daytona_workspace_cache: dict[str, str] = {}
_state_backend = StateBackend()
_thread_scoped_sandbox_backend: "ThreadScopedSandboxBackend | None" = None


def daytona_sandbox_enabled() -> bool:
    """Return True when Daytona should back the deepagents filesystem."""
    return os.getenv("DAYTONA_SANDBOX_ENABLED", "false").lower() in (
        "1",
        "true",
        "yes",
    )


def filesystem_backend() -> BackendProtocol:
    """Return a ``BackendProtocol`` instance for ``create_deep_agent``.

    deepagents 0.7+ deprecates passing a callable backend factory. We pass a
    concrete instance instead: ``StateBackend`` locally, or ``ThreadScopedSandboxBackend``
    when Daytona is enabled (resolves the thread sandbox on each file operation).
    """
    if daytona_sandbox_enabled():
        global _thread_scoped_sandbox_backend
        if _thread_scoped_sandbox_backend is None:
            _thread_scoped_sandbox_backend = ThreadScopedSandboxBackend()
        return _thread_scoped_sandbox_backend
    return _state_backend


def _get_daytona_client() -> Daytona:
    """Return a process-wide Daytona client."""
    global _daytona_client_instance
    if _daytona_client_instance is None:
        api_key = os.getenv("DAYTONA_API_KEY")
        if not api_key:
            raise ValueError("DAYTONA_API_KEY is required for Daytona sandbox backend")
        _daytona_client_instance = Daytona(DaytonaConfig(api_key=api_key))
    return _daytona_client_instance


def _thread_id_from_config() -> str:
    """Read ``thread_id`` from the active LangGraph run config."""
    thread_id = get_config().get("configurable", {}).get("thread_id")
    if thread_id:
        return str(thread_id)
    raise ValueError(
        "thread_id is required in config['configurable'] for sandbox backend"
    )


def _ensure_sandbox_started(client: Daytona, sandbox: Sandbox) -> None:
    """Start a stopped Daytona sandbox before use."""
    if sandbox.state == SandboxState.STARTED:
        return
    client.start(sandbox)


def _resolve_daytona_sandbox(thread_id: str) -> Sandbox:
    """Return an existing thread-scoped Daytona sandbox or create one."""
    sandbox_name = f"thread-{thread_id}"
    client = _get_daytona_client()

    try:
        sandbox = client.get(sandbox_name)
    except DaytonaNotFoundError:
        # Daytona ``auto_stop_interval`` is idle time in *minutes*, not seconds.
        auto_stop_interval = int(
            os.getenv(
                "DAYTONA_SANDBOX_AUTO_STOP_INTERVAL_SECONDS",
                str(DEFAULT_SANDBOX_AUTO_STOP_INTERVAL_SECONDS),
            )
        )
        sandbox = client.create(
            CreateSandboxFromSnapshotParams(
                name=sandbox_name,
                auto_stop_interval=auto_stop_interval,
                labels={
                    "app": "deep-agents-from-scratch",
                    "thread_id": thread_id,
                },
            )
        )
    else:
        _ensure_sandbox_started(client, sandbox)

    return sandbox


def _resolve_thread_backend() -> BackendProtocol:
    """Resolve the filesystem backend for the current LangGraph thread."""
    if not daytona_sandbox_enabled():
        return _state_backend

    thread_id = _thread_id_from_config()
    try:
        sandbox = _resolve_daytona_sandbox(thread_id)
        return build_daytona_backend(sandbox, thread_id)
    except Exception as exc:
        logger.warning(
            "Daytona sandbox unavailable for thread %s; using StateBackend: %s",
            thread_id,
            exc,
        )
        return _state_backend


class ThreadScopedSandboxBackend(SandboxBackendProtocol):
    """``BackendProtocol`` instance that picks a per-thread Daytona sandbox at runtime.

    Replaces the deprecated ``backend=get_sandbox`` factory passed to
    ``create_deep_agent``. Each filesystem tool call reads ``thread_id`` from
    LangGraph config and delegates to ``PrefixedSandboxBackend`` for that thread.
    """

    def _inner(self) -> BackendProtocol:
        return _resolve_thread_backend()

    @property
    def id(self) -> str:
        inner = self._inner()
        if not isinstance(inner, SandboxBackendProtocol):
            msg = "Sandbox id is unavailable without an active Daytona backend"
            raise NotImplementedError(msg)
        return inner.id

    def ls(self, path: str) -> LsResult:
        return self._inner().ls(path)

    async def als(self, path: str) -> LsResult:
        return await asyncio.to_thread(self.ls, path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self._inner().read(file_path, offset=offset, limit=limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        return self._inner().write(file_path, content)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return await asyncio.to_thread(self.write, file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return self._inner().edit(
            file_path, old_string, new_string, replace_all=replace_all
        )

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return await asyncio.to_thread(
            self.edit, file_path, old_string, new_string, replace_all
        )

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        return self._inner().grep(pattern, path, glob)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        return await asyncio.to_thread(self.grep, pattern, path, glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        return self._inner().glob(pattern, path)

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        return await asyncio.to_thread(self.glob, pattern, path)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._inner().download_files(paths)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return await asyncio.to_thread(self.download_files, paths)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._inner().upload_files(files)

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        return await asyncio.to_thread(self.upload_files, files)

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        inner = self._inner()
        if not isinstance(inner, SandboxBackendProtocol):
            msg = (
                "Execution not available. Daytona is disabled or unavailable; "
                "use DAYTONA_SANDBOX_ENABLED=true with a valid DAYTONA_API_KEY."
            )
            raise NotImplementedError(msg)
        if timeout is not None:
            return inner.execute(command, timeout=timeout)
        return inner.execute(command)

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        return await asyncio.to_thread(self.execute, command, timeout=timeout)


# Base directory inside the sandbox for all thread workspaces.
# When unset, ``sandbox.get_work_dir()`` is used at runtime (Daytona's writable cwd).


def daytona_workspace_root(sandbox: Sandbox) -> str:
    """Return the sandbox-wide workspace root (parent of per-thread dirs)."""
    configured = os.getenv("DAYTONA_WORKSPACE_ROOT", "").strip()
    if configured:
        return configured.rstrip("/")

    # Option 3: use Daytona's declared work directory instead of hard-coding /home/...
    return sandbox.get_work_dir().rstrip("/")


def thread_physical_root(sandbox: Sandbox, thread_id: str) -> str:
    """Physical path where this thread's virtual ``/`` is stored in the sandbox."""
    workspace_root = daytona_workspace_root(sandbox)
    threads_subdir = os.getenv(
        "DAYTONA_WORKSPACE_THREADS_SUBDIR", DEFAULT_WORKSPACE_THREADS_SUBDIR
    ).strip("/")
    return f"{workspace_root}/{threads_subdir}/{thread_id}"


def ensure_thread_workspace(backend: DaytonaSandbox, physical_root: str) -> None:
    """Create the per-thread workspace directory inside the sandbox (idempotent)."""
    quoted = shlex.quote(physical_root)
    result = backend.execute(f"mkdir -p {quoted}", timeout=60)
    if result.exit_code != 0:
        msg = (result.output or "").strip() or f"exit code {result.exit_code}"
        raise RuntimeError(f"Failed to create Daytona workspace {physical_root}: {msg}")


class PrefixedSandboxBackend(SandboxBackendProtocol):
    """Maps deepagents virtual paths (``/foo.txt``) to a sandbox workspace prefix.

    Option 1: the agent and prompts stay unchanged; path translation happens here
    on every file operation so Daytona never receives writes to filesystem ``/``.
    """

    def __init__(self, inner: DaytonaSandbox, *, physical_root: str) -> None:
        self._inner = inner
        # e.g. /home/daytona/workspace/threads/<thread_id>
        self._physical_root = physical_root.rstrip("/")

    @property
    def id(self) -> str:
        return self._inner.id

    def _to_physical(self, virtual_path: str) -> str:
        """Translate agent virtual path → writable path inside the sandbox."""
        if not virtual_path.startswith("/"):
            raise ValueError(f"Path must be absolute, got {virtual_path!r}")
        if virtual_path == "/":
            return self._physical_root
        return f"{self._physical_root}{virtual_path}"

    def _to_virtual(self, physical_path: str) -> str:
        """Translate sandbox path → agent virtual path for tool responses."""
        root = self._physical_root
        if physical_path == root:
            return "/"
        prefix = f"{root}/"
        if physical_path.startswith(prefix):
            return physical_path[len(root) :]
        return physical_path

    def _remap_file_info(self, entry: FileInfo) -> FileInfo:
        return cast(
            FileInfo,
            {**entry, "path": self._to_virtual(entry["path"])},
        )

    def _remap_grep_match(self, match: GrepMatch) -> GrepMatch:
        return cast(
            GrepMatch,
            {**match, "path": self._to_virtual(match["path"])},
        )

    def ls(self, path: str) -> LsResult:
        result = self._inner.ls(self._to_physical(path))
        if result.entries is None:
            return result
        return LsResult(
            error=result.error,
            entries=[self._remap_file_info(entry) for entry in result.entries],
        )

    async def als(self, path: str) -> LsResult:
        return await asyncio.to_thread(self.ls, path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self._inner.read(self._to_physical(file_path), offset=offset, limit=limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        physical_path = self._to_physical(file_path)
        result = self._inner.write(physical_path, content)
        if result.error is not None:
            return result
        return WriteResult(path=self._to_virtual(result.path or physical_path))

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return await asyncio.to_thread(self.write, file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        physical_path = self._to_physical(file_path)
        result = self._inner.edit(
            physical_path,
            old_string,
            new_string,
            replace_all=replace_all,
        )
        if result.error is not None:
            return result
        return EditResult(
            path=self._to_virtual(result.path or physical_path),
            occurrences=result.occurrences,
        )

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return await asyncio.to_thread(
            self.edit, file_path, old_string, new_string, replace_all
        )

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        physical_path = self._to_physical(path) if path is not None else None
        result = self._inner.grep(pattern, physical_path, glob)
        if result.matches is None:
            return result
        return GrepResult(
            error=result.error,
            matches=[self._remap_grep_match(match) for match in result.matches],
        )

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        return await asyncio.to_thread(self.grep, pattern, path, glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        result = self._inner.glob(pattern, self._to_physical(path))
        if result.matches is None:
            return result
        return GlobResult(
            error=result.error,
            matches=[self._remap_file_info(match) for match in result.matches],
        )

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        return await asyncio.to_thread(self.glob, pattern, path)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        physical_paths = [self._to_physical(path) for path in paths]
        responses = self._inner.download_files(physical_paths)
        return [
            FileDownloadResponse(
                path=self._to_virtual(response.path),
                content=response.content,
                error=response.error,
            )
            for response in responses
        ]

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return await asyncio.to_thread(self.download_files, paths)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        physical_files = [
            (self._to_physical(path), content) for path, content in files
        ]
        responses = self._inner.upload_files(physical_files)
        return [
            FileUploadResponse(
                path=self._to_virtual(response.path),
                error=response.error,
            )
            for response in responses
        ]

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        return await asyncio.to_thread(self.upload_files, files)

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        # Shell commands from BaseSandbox already embed physical paths because
        # file ops above pass translated paths to the inner backend.
        return self._inner.execute(command, timeout=timeout)

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        return await asyncio.to_thread(self.execute, command, timeout=timeout)


def build_daytona_backend(sandbox: Sandbox, thread_id: str) -> BackendProtocol:
    """Provision workspace and return a path-prefixed Daytona backend for a thread."""
    cache_key = f"{sandbox.id}:{thread_id}"
    if cache_key in _daytona_workspace_cache:
        physical_root = _daytona_workspace_cache[cache_key]
    else:
        physical_root = thread_physical_root(sandbox, thread_id)
        inner = DaytonaSandbox(sandbox=sandbox)
        # Option 3: ensure the thread workspace exists before any agent file writes.
        ensure_thread_workspace(inner, physical_root)
        _daytona_workspace_cache[cache_key] = physical_root

    inner = DaytonaSandbox(sandbox=sandbox)
    return PrefixedSandboxBackend(inner, physical_root=physical_root)
