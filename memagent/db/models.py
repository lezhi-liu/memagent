"""ORM models. The migration in `migrations/versions/0001_s1_initial.py` is the source of truth
for constraints and the events trigger; these classes mirror it for typed queries.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKeyConstraint,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

type EventType = Literal["user_message", "assistant_message", "tool_call", "tool_result", "error"]
type Module = Literal["agent", "search"]

EVENT_TYPES: tuple[str, ...] = (
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "error",
)
REDACTED_PAYLOAD: dict[str, Any] = {"redacted": True}


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChatSession(Base):
    """A conversation. Named ChatSession so it never gets confused with a DB session."""

    __tablename__ = "sessions"
    __table_args__ = (UniqueConstraint("id", "user_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("session_id", "seq"),
        ForeignKeyConstraint(["session_id", "user_id"], ["sessions.id", "sessions.user_id"]),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID)
    seq: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    redacted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class LLMCall(Base):
    __tablename__ = "llm_calls"
    __table_args__ = (
        ForeignKeyConstraint(["session_id", "user_id"], ["sessions.id", "sessions.user_id"]),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID)
    module: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    search_calls: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
