"""Per-user workspace files: `<workspace_root>/<user_id>/`.

Two protections:
- Path traversal: the requested path is resolved (following `..` and symlinks) and must
  still be inside the user's own root. The user id comes from ToolContext, never from
  the model, so the model cannot name another user's directory.
- Scrubbing: content is scrubbed before it is written, like every other persisted text.
"""

import uuid
from pathlib import Path
from typing import Any

from memagent.llm.types import ToolSpec
from memagent.sanitize import scrub
from memagent.tools.base import ToolContext, ToolError, ToolOutput, require_str

MAX_FILE_BYTES = 256 * 1024
MAX_LIST_ENTRIES = 500


def user_root(workspace_root: Path, user_id: uuid.UUID) -> Path:
    root = (workspace_root / str(user_id)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_in_workspace(workspace_root: Path, user_id: uuid.UUID, rel_path: str) -> Path:
    """Map a model-supplied relative path to a real path inside the user's root, or refuse.

    Checking the string for ".." is not enough (symlinks, absolute paths, odd encodings);
    resolving first and then comparing real paths is.
    """
    root = user_root(workspace_root, user_id)
    if Path(rel_path).is_absolute():
        raise ToolError("path must be relative to your workspace")
    target = (root / rel_path).resolve()
    if not target.is_relative_to(root):
        raise ToolError("path is outside your workspace")
    return target


class WorkspaceListTool:
    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="workspace_list",
            description="List the files in your private workspace (paths relative to it).",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolOutput:
        root = user_root(self._root, ctx.user_id)
        files = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
        if not files:
            return ToolOutput("(workspace is empty)")
        shown = files[:MAX_LIST_ENTRIES]
        more = len(files) - len(shown)
        return ToolOutput("\n".join(shown) + (f"\n... and {more} more" if more else ""))


class WorkspaceReadTool:
    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="workspace_read",
            description="Read a UTF-8 text file from your private workspace.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative path"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolOutput:
        path = resolve_in_workspace(self._root, ctx.user_id, require_str(args, "path"))
        if not path.is_file():
            raise ToolError("file not found")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ToolError(f"file is larger than {MAX_FILE_BYTES} bytes")
        try:
            return ToolOutput(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise ToolError("file is not UTF-8 text") from exc


class WorkspaceWriteTool:
    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="workspace_write",
            description=(
                "Create or overwrite a UTF-8 text file in your private workspace. "
                "Secrets such as passwords and API keys are redacted before saving."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolOutput:
        path = resolve_in_workspace(self._root, ctx.user_id, require_str(args, "path"))
        content = args.get("content")
        if not isinstance(content, str):
            raise ToolError("argument 'content' is required and must be a string")
        clean = scrub(content)
        data = clean.text.encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            raise ToolError(f"content is larger than {MAX_FILE_BYTES} bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        redacted = sum(clean.counts.values())
        note = f" ({redacted} sensitive value(s) redacted)" if redacted else ""
        return ToolOutput(f"wrote {len(data)} bytes to {args['path']}{note}")
