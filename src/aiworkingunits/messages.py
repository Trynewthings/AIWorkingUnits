from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str | None = None
    capability: str | None = None
    type: MessageType = MessageType.REQUEST
    correlation_id: str | None = None
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    def reply(self, payload: dict[str, Any], *, sender: str, type: MessageType = MessageType.RESPONSE) -> Message:
        return Message(
            sender=sender,
            receiver=self.sender,
            type=type,
            correlation_id=self.id,
            trace_id=self.trace_id,
            payload=payload,
        )
