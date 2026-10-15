"""The scrubbing gateway: unit tests per rule, plus end-to-end "nothing raw reaches storage"."""

import uuid
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from memagent.db.engine import SessionMaker
from memagent.llm.fake import FakeLLM, reply
from memagent.llm.types import UserMessage
from memagent.sanitize import scrub, scrub_json
from memagent.tools.workspace import WorkspaceWriteTool
from tests.conftest import MakeClient, MakeUser, tool_context

# Synthetic values only. The ID and card numbers satisfy their check digits.
OPENAI_KEY = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
VALID_CN_ID = "11010519491231002X"
VALID_CARD = "4111111111111111"
PASSWORD = "hunter2-Secret!"


@pytest.mark.parametrize(
    "secret",
    [
        OPENAI_KEY,
        "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_" + "a1" * 18,
        "AIza" + "B" * 35,
        "xoxb-1234567890-abcdefghij",
        "mk_" + "x" * 43,
    ],
)
def test_api_key_formats(secret: str) -> None:
    result = scrub(f"here is my key {secret} thanks")
    assert secret not in result.text
    assert result.counts == {"api_key": 1}


@pytest.mark.parametrize(
    ("text_in", "value"),
    [
        ("password=hunter22", "hunter22"),
        ("my password is hunter22", "hunter22"),
        ("token: abc123xyz", "abc123xyz"),
        ("client_secret = 's3cr3t value'", "s3cr3t value"),
        ('{"api_key": "zzz999"}', "zzz999"),
        ("密码：abc123", "abc123"),
        ("我的密码是abc123，记住", "abc123"),
        ("口令为 qwerty", "qwerty"),
        ("令牌: tok_12345", "tok_12345"),
        ("密钥=k3y", "k3y"),
    ],
)
def test_keyword_values(text_in: str, value: str) -> None:
    result = scrub(text_in)
    assert value not in result.text
    assert "[REDACTED:secret]" in result.text
    assert result.counts == {"secret": 1}


def test_keyword_is_kept_only_value_removed() -> None:
    assert scrub("我的密码是abc123，记住").text == "我的密码是[REDACTED:secret]，记住"


def test_pem_private_key_block() -> None:
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\nIBAAKC\n-----END RSA PRIVATE KEY-----"
    result = scrub(f"key:\n{pem}\nend")
    assert "MIIEow" not in result.text
    assert result.counts == {"private_key": 1}


def test_bank_card_requires_luhn() -> None:
    assert scrub(f"card {VALID_CARD}").counts == {"bank_card": 1}
    assert scrub("card 4111 1111 1111 1111").counts == {"bank_card": 1}
    # Off by one digit: fails Luhn, so it is left alone (e.g. an order number).
    assert scrub("order 4111111111111112").counts == {}


def test_cn_id_requires_check_digit() -> None:
    result = scrub(f"身份证 {VALID_CN_ID}")
    assert VALID_CN_ID not in result.text
    assert result.counts == {"cn_id": 1}
    assert scrub("身份证 110105194912310021").counts == {}  # wrong check digit


def test_counts_add_up_and_clean_text_untouched() -> None:
    assert scrub("nothing sensitive here").counts == {}
    result = scrub(f"password: {PASSWORD} key {OPENAI_KEY} id {VALID_CN_ID}")
    assert result.counts == {"secret": 1, "api_key": 1, "cn_id": 1}


def test_scrub_json_walks_nested_values() -> None:
    clean, counts = scrub_json({"a": [{"b": f"key {OPENAI_KEY}"}], "n": 3, "ok": True})
    assert clean == {"a": [{"b": "key [REDACTED:api_key]"}], "n": 3, "ok": True}
    assert counts == {"api_key": 1}


# ---- end to end -------------------------------------------------------------------------

SENSITIVE_MESSAGE = f"my password is {PASSWORD} and my key is {OPENAI_KEY}, ID {VALID_CN_ID}"


async def test_raw_values_never_reach_the_database(
    make_user: MakeUser, make_client: MakeClient, engine: AsyncEngine
) -> None:
    _, key = await make_user()
    llm = FakeLLM([reply("noted")])
    client = await make_client(llm, key)
    session_id = (await client.post("/sessions")).json()["id"]
    response = await client.post(
        f"/sessions/{session_id}/messages", json={"text": SENSITIVE_MESSAGE}
    )
    assert response.status_code == 200

    async with engine.connect() as conn:
        stored = await conn.scalar(
            text("SELECT string_agg(payload::text, ' ') FROM events WHERE session_id = :sid"),
            {"sid": uuid.UUID(session_id)},
        )
    assert stored is not None
    for raw in (PASSWORD, OPENAI_KEY, VALID_CN_ID):
        assert raw not in stored

    # The model only ever saw the scrubbed version (history is rebuilt from the DB).
    user_messages = [m for m in llm.calls[0] if isinstance(m, UserMessage)]
    assert PASSWORD not in user_messages[0].text
    assert "[REDACTED:secret]" in user_messages[0].text


async def test_workspace_write_is_scrubbed(tmp_path: Path, sessionmaker: SessionMaker) -> None:
    tool = WorkspaceWriteTool(tmp_path)
    ctx = tool_context(sessionmaker)
    await tool.run(ctx, {"path": "notes.txt", "content": SENSITIVE_MESSAGE})

    on_disk = (tmp_path / str(ctx.user_id) / "notes.txt").read_text(encoding="utf-8")
    for raw in (PASSWORD, OPENAI_KEY, VALID_CN_ID):
        assert raw not in on_disk
