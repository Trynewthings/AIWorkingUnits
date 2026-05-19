from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from aiworkingunits.bus import AsyncMessageBus
from aiworkingunits.messages import Message, MessageType

logger = logging.getLogger(__name__)


class UnitConfig(BaseModel):
    """Base configuration for a Working Unit.

    Subclass per unit to add business parameters (model name, prompts, paths,
    thresholds, etc.). Two units sharing the same code but different configs
    is the supported way to "reuse a unit for a similar task".
    """

    unit_id: str
    capabilities: list[str] = Field(default_factory=list)
    model: str = "claude-sonnet-4-5"
    temperature: float = 0.2
    max_tokens: int = 4096
    request_timeout_s: float = 60.0


class WorkingUnit(ABC):
    config_cls: type[UnitConfig] = UnitConfig

    def __init__(self, config: UnitConfig, bus: AsyncMessageBus) -> None:
        self.config = config
        self.bus = bus
        self.inbox: asyncio.Queue[Message] | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    @property
    def unit_id(self) -> str:
        return self.config.unit_id

    async def start(self) -> None:
        self.inbox = self.bus.register(self.unit_id, self.config.capabilities)
        self._task = asyncio.create_task(self._run(), name=f"unit:{self.unit_id}")
        logger.info("unit started id=%s", self.unit_id)

    async def stop(self) -> None:
        self._stopping.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self.bus.unregister(self.unit_id)
        logger.info("unit stopped id=%s", self.unit_id)

    async def _run(self) -> None:
        assert self.inbox is not None
        while not self._stopping.is_set():
            msg = await self.inbox.get()
            try:
                reply = await self.handle(msg)
                if reply is not None:
                    await self.bus.publish(reply)
            except Exception as e:
                logger.exception("unit=%s failed handling msg id=%s", self.unit_id, msg.id)
                if msg.type == MessageType.REQUEST:
                    err = msg.reply(
                        payload={"error": str(e), "error_type": type(e).__name__},
                        sender=self.unit_id,
                        type=MessageType.ERROR,
                    )
                    await self.bus.publish(err)

    @abstractmethod
    async def handle(self, msg: Message) -> Message | None:
        """Process an inbound message. Return a response Message or None."""

    async def send_event(self, capability: str, payload: dict[str, Any]) -> None:
        await self.bus.publish(
            Message(sender=self.unit_id, capability=capability, type=MessageType.EVENT, payload=payload)
        )

    async def request(self, *, receiver: str | None = None, capability: str | None = None, payload: dict[str, Any], timeout: float | None = None) -> Message:
        msg = Message(
            sender=self.unit_id,
            receiver=receiver,
            capability=capability,
            type=MessageType.REQUEST,
            payload=payload,
        )
        return await self.bus.request(msg, timeout=timeout or self.config.request_timeout_s)
