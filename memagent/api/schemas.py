"""Request and response bodies."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


class SessionOut(BaseModel):
    id: uuid.UUID
    created_at: datetime


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=32_000)


class StepOut(BaseModel):
    tool: str
    is_error: bool


class TurnOut(BaseModel):
    reply: str
    steps: list[StepOut]
    cost_usd: Decimal
    stopped_by_limit: bool


class EventOut(BaseModel):
    id: int
    seq: int
    type: str
    payload: dict[str, Any]
    created_at: datetime
    redacted_at: datetime | None
