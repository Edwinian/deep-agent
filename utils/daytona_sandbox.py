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
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)
from langchain_daytona import DaytonaSandbox
from langgraph.config import get_config

from db.agent_store import SkillRow

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_AUTO_STOP_INTERVAL_MINUTES = 30
DEFAULT_SANDBOX_AUTO_DELETE_INTERVAL_MINUTES = 10080
DEFAULT_WORKSPACE_THREADS_SUBDIR = "threads"

_daytona_client_instance: Daytona | None = None
_daytona_workspace_cache: dict[str, str] = {}
_thread_scoped_sandbox_backend: "ThreadScopedSandboxBackend | None" = None


def daytona_sandbox_enabled() -> bool:
    """Return True when Daytona should back the deepagents filesystem."""
    return os.getenv("DAYTONA_SANDBOX_ENABLED", "false").lower() in (
        "1",
        "true",
        "yes",
    )


def filesystem_backend():
    """Return a backend instance or factory for ``create_deep_agent``.

    deepagents 0.3.x expects either a ``BackendProtocol`` or a
    ``BackendFactory`` (``Callable[[ToolRuntime], BackendProtocol]``).
    ``StateBackend`` requires a ``ToolRuntime``, so the non-Daytona path
    returns a factory. Daytona uses a thread-scoped instance that resolves
    the sandbox on each file operation.
    """
    if daytona_sandbox_enabled():
        global _thread_scoped_sandbox_backend
        if _thread_scoped_sandbox_backend is None:
            _thread_scoped_sandbox_backend = ThreadScopedSandboxBackend()
        return _thread_scoped_sandbox_backend
    return lambda runtime: StateBackend(runtime)


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


def _sandbox_name(thread_id: str) -> str:
    return f"thread-{thread_id}"


def _sandbox_auto_stop_interval_minutes() -> int:
    """Idle minutes before Daytona auto-stops a sandbox."""
    return int(
        os.getenv(
            "DAYTONA_SANDBOX_AUTO_STOP_INTERVAL_MINUTES",
            str(DEFAULT_SANDBOX_AUTO_STOP_INTERVAL_MINUTES),
        )
    )


def _sandbox_auto_delete_interval_minutes() -> int | None:
    """Idle minutes before Daytona auto-deletes a sandbox (None disables)."""
    raw = os.getenv(
        "DAYTONA_SANDBOX_AUTO_DELETE_INTERVAL_MINUTES",
        str(DEFAULT_SANDBOX_AUTO_DELETE_INTERVAL_MINUTES),
    ).strip()
    if not raw or raw.lower() in {"0", "none", "false", "off"}:
        return None
    return int(raw)


def delete_daytona_sandbox(thread_id: str) -> bool:
    """Delete the Daytona sandbox for a thread, if it exists."""
    if not daytona_sandbox_enabled():
        return False

    client = _get_daytona_client()
    sandbox_name = _sandbox_name(thread_id)
    try:
        sandbox = client.get(sandbox_name)
    except DaytonaNotFoundError:
        return False

    client.delete(sandbox)
    suffix = f":{thread_id}"
    stale_keys = [key for key in _daytona_workspace_cache if key.endswith(suffix)]
    for key in stale_keys:
        del _daytona_workspace_cache[key]
    return True


def _ensure_sandbox_started(client: Daytona, sandbox: Sandbox) -> None:
    """Start a stopped Daytona sandbox before use."""
    if sandbox.state == SandboxState.STARTED:
        return
    client.start(sandbox)


def _resolve_daytona_sandbox(thread_id: str) -> Sandbox:
    """Return an existing thread-scoped Daytona sandbox or create one."""
    sandbox_name = _sandbox_name(thread_id)
    client = _get_daytona_client()

    try:
        sandbox = client.get(sandbox_name)
    except DaytonaNotFoundError:
        create_params: dict[str, object] = {
            "name": sandbox_name,
            "auto_stop_interval": _sandbox_auto_stop_interval_minutes(),
            "labels": {
                "app": "deep-agents-from-scratch",
                "thread_id": thread_id,
            },
        }
        auto_delete_interval = _sandbox_auto_delete_interval_minutes()
        if auto_delete_interval is not None:
            create_params["auto_delete_interval"] = auto_delete_interval
        sandbox = client.create(CreateSandboxFromSnapshotParams(**create_params))
    else:
        _ensure_sandbox_started(client, sandbox)

    return sandbox


def _resolve_thread_backend() -> BackendProtocol:
    """Resolve the filesystem backend for the current LangGraph thread."""
    if not daytona_sandbox_enabled():
        raise RuntimeError(
            "Daytona is disabled; ThreadScopedSandboxBackend should not be used. "
            "Pass filesystem_backend()'s StateBackend factory instead."
        )

    thread_id = _thread_id_from_config()
    sandbox = _resolve_daytona_sandbox(thread_id)
    return build_daytona_backend(sandbox, thread_id)


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

    def ls_info(self, path: str) -> list[FileInfo]:
        return self._inner().ls_info(path)

    async def als_info(self, path: str) -> list[FileInfo]:
        return await asyncio.to_thread(self.ls_info, path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return self._inner().read(file_path, offset=offset, limit=limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
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

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        return self._inner().grep_raw(pattern, path, glob)

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        return await asyncio.to_thread(self.grep_raw, pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return self._inner().glob_info(pattern, path)

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return await asyncio.to_thread(self.glob_info, pattern, path)

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
        del timeout  # deepagents 0.3 / langchain-daytona execute(command) only
        inner = self._inner()
        if not isinstance(inner, SandboxBackendProtocol):
            msg = (
                "Execution not available. Daytona is disabled or unavailable; "
                "use DAYTONA_SANDBOX_ENABLED=true with a valid DAYTONA_API_KEY."
            )
            raise NotImplementedError(msg)
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
    result = backend.execute(f"mkdir -p {quoted}")
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

    def ls_info(self, path: str) -> list[FileInfo]:
        entries = self._inner.ls_info(self._to_physical(path))
        return [self._remap_file_info(entry) for entry in entries]

    async def als_info(self, path: str) -> list[FileInfo]:
        return await asyncio.to_thread(self.ls_info, path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return self._inner.read(self._to_physical(file_path), offset=offset, limit=limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
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

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        physical_path = self._to_physical(path) if path is not None else None
        result = self._inner.grep_raw(pattern, physical_path, glob)
        if isinstance(result, str):
            return result
        return [self._remap_grep_match(match) for match in result]

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        return await asyncio.to_thread(self.grep_raw, pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        matches = self._inner.glob_info(pattern, self._to_physical(path))
        return [self._remap_file_info(match) for match in matches]

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return await asyncio.to_thread(self.glob_info, pattern, path)

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
        del timeout  # deepagents 0.3 / langchain-daytona execute(command) only
        # Shell commands from BaseSandbox already embed physical paths because
        # file ops above pass translated paths to the inner backend.
        return self._inner.execute(command)

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


SKILLS_ROOT = "/skills"


def _skill_file_path(skill_id: int) -> str:
    return f"{SKILLS_ROOT}/skill_{skill_id}/SKILL.md"


def _skills_root_paths() -> list[str]:
    return [f"{SKILLS_ROOT}/"]


def _backend_for_skills(*, thread_id: str | None) -> BackendProtocol:
    if not daytona_sandbox_enabled():
        raise RuntimeError(
            "Skill file sync requires Daytona when using deepagents 0.3 StateBackend "
            "(ToolRuntime is unavailable at skill-load time). "
            "Set DAYTONA_SANDBOX_ENABLED=true or skip skill file materialization."
        )
    if thread_id is None:
        msg = "thread_id is required to sync skills to a Daytona sandbox"
        raise ValueError(msg)
    sandbox = _resolve_daytona_sandbox(thread_id)
    return build_daytona_backend(sandbox, thread_id)


def _ensure_parent_dir(backend: BackendProtocol, file_path: str) -> None:
    parent = file_path.rsplit("/", 1)[0]
    if parent in {"", "/"}:
        return
    if isinstance(backend, PrefixedSandboxBackend):
        physical_parent = backend._to_physical(parent)
        quoted = shlex.quote(physical_parent)
        result = backend._inner.execute(f"mkdir -p {quoted}")
    elif isinstance(backend, SandboxBackendProtocol):
        quoted = shlex.quote(parent)
        result = backend.execute(f"mkdir -p {quoted}")
    else:
        return
    if result.exit_code != 0:
        msg = (result.output or "").strip() or f"exit code {result.exit_code}"
        raise RuntimeError(f"Failed to create skill directory {parent}: {msg}")


def load_skills(
    skill_rows: list[SkillRow],
    *,
    thread_id: str | None = None,
) -> list[str]:
    """Write skills into the active backend and return skill source paths.

    Each skill is stored as ``/skills/skill_<id>/SKILL.md``. When Daytona is
    enabled, ``thread_id`` is required to provision files in the thread sandbox.
    Without a thread id, only the virtual skill root path is returned so agents
    can be compiled before a run starts.
    """
    if not skill_rows:
        return []

    # Without Daytona, StateBackend needs ToolRuntime (unavailable at compile time).
    # Return skill roots only; runtime SkillsMiddleware / thread sync handles content.
    if not daytona_sandbox_enabled():
        return _skills_root_paths()

    if thread_id is None:
        return _skills_root_paths()

    backend = _backend_for_skills(thread_id=thread_id)
    for row in skill_rows:
        skill_file = _skill_file_path(row.id)
        _ensure_parent_dir(backend, skill_file)
        result = backend.write(skill_file, row.content)
        if result.error is not None:
            raise RuntimeError(
                f"Failed to write skill {row.id} to {skill_file}: {result.error}"
            )

    return _skills_root_paths()


def sync_skills_for_thread(agent_id: int, thread_id: str) -> list[str]:
    """Load an agent's configured skills from SQLite into the thread sandbox."""
    from db.agent_store import get_agent, get_skills

    row = get_agent(agent_id)
    if not row.skill_ids:
        return []
    return load_skills(get_skills(row.skill_ids), thread_id=thread_id)
