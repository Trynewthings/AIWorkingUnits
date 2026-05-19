from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from aiworkingunits.messages import Message, MessageType

logger = logging.getLogger(__name__)


class AsyncMessageBus:
    """In-process message bus for Working Unit communication.

    Routing: by explicit receiver unit_id, or by capability (broadcast to all
    units that advertised it). Request/response is paired via correlation_id.
    """

    def __init__(self) -> None:
        self._inboxes: dict[str, asyncio.Queue[Message]] = {}
        self._capabilities: dict[str, list[str]] = defaultdict(list)
        self._pending: dict[str, asyncio.Future[Message]] = {}

    def register(self, unit_id: str, capabilities: list[str]) -> asyncio.Queue[Message]:
        if unit_id in self._inboxes:
            raise ValueError(f"unit_id already registered: {unit_id}")
        queue: asyncio.Queue[Message] = asyncio.Queue()
        self._inboxes[unit_id] = queue
        for cap in capabilities:
            self._capabilities[cap].append(unit_id)
        logger.info("registered unit=%s capabilities=%s", unit_id, capabilities)
        return queue

    def unregister(self, unit_id: str) -> None:
        self._inboxes.pop(unit_id, None)
        for unit_list in self._capabilities.values():
            if unit_id in unit_list:
                unit_list.remove(unit_id)

    def resolve(self, msg: Message) -> list[str]:
        if msg.receiver:
            return [msg.receiver] if msg.receiver in self._inboxes else []
        if msg.capability:
            return list(self._capabilities.get(msg.capability, []))
        return []

    async def publish(self, msg: Message) -> None:
        # Responses always go straight back to the original sender (the
        # original message's receiver), regardless of any capability hints.
        if msg.type == MessageType.RESPONSE and msg.correlation_id in self._pending:
            future = self._pending.pop(msg.correlation_id)
            if not future.done():
                future.set_result(msg)
            return

        targets = self.resolve(msg)
        if not targets:
            logger.warning("no target for message id=%s receiver=%s capability=%s", msg.id, msg.receiver, msg.capability)
            return
        for unit_id in targets:
            await self._inboxes[unit_id].put(msg)

    async def request(self, msg: Message, timeout: float = 60.0) -> Message:
        if msg.type != MessageType.REQUEST:
            raise ValueError("request() requires a REQUEST-type message")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Message] = loop.create_future()
        self._pending[msg.id] = future
        await self.publish(msg)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(msg.id, None)
