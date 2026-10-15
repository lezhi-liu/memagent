"""Workspace tools: round trip, path traversal, size limit."""

import uuid
from pathlib import Path

import pytest

from memagent.db.engine import SessionMaker
from memagent.tools.base import ToolError
from memagent.tools.workspace import (
    MAX_FILE_BYTES,
    WorkspaceListTool,
    WorkspaceReadTool,
    WorkspaceWriteTool,
    resolve_in_workspace,
)
from tests.conftest import tool_context


async def test_write_read_list_round_trip(tmp_path: Path, sessionmaker: SessionMaker) -> None:
    ctx = tool_context(sessionmaker)
    await WorkspaceWriteTool(tmp_path).run(ctx, {"path": "notes/todo.md", "content": "buy milk"})

    read = await WorkspaceReadTool(tmp_path).run(ctx, {"path": "notes/todo.md"})
    listing = await WorkspaceListTool(tmp_path).run(ctx, {})

    assert read.text == "buy milk"
    assert listing.text == "notes/todo.md"


@pytest.mark.parametrize("bad_path", ["../escape.txt", "a/../../escape.txt", "/etc/passwd"])
def test_traversal_is_rejected(tmp_path: Path, bad_path: str) -> None:
    with pytest.raises(ToolError):
        resolve_in_workspace(tmp_path, uuid.uuid4(), bad_path)


def test_other_users_directory_is_rejected(tmp_path: Path) -> None:
    me, other = uuid.uuid4(), uuid.uuid4()
    (tmp_path / str(other)).mkdir()
    with pytest.raises(ToolError):
        resolve_in_workspace(tmp_path, me, f"../{other}/secret.txt")


def test_symlink_escaping_the_root_is_rejected(tmp_path: Path) -> None:
    user_id = uuid.uuid4()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not yours")
    root = tmp_path / str(user_id)
    root.mkdir()
    (root / "link").symlink_to(outside)

    with pytest.raises(ToolError):
        resolve_in_workspace(tmp_path, user_id, "link/secret.txt")


async def test_traversal_through_the_tool_is_an_error(
    tmp_path: Path, sessionmaker: SessionMaker
) -> None:
    ctx = tool_context(sessionmaker)
    with pytest.raises(ToolError):
        await WorkspaceReadTool(tmp_path).run(ctx, {"path": "../../etc/passwd"})


async def test_size_limit(tmp_path: Path, sessionmaker: SessionMaker) -> None:
    ctx = tool_context(sessionmaker)
    with pytest.raises(ToolError, match="larger than"):
        await WorkspaceWriteTool(tmp_path).run(
            ctx, {"path": "big.txt", "content": "x" * (MAX_FILE_BYTES + 1)}
        )
