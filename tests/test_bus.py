from __future__ import annotations

import asyncio

import pytest

from aiworkingunits import AsyncMessageBus, Message, MessageType, UnitConfig, WorkingUnit


class EchoUnit(WorkingUnit):
    async def handle(self, msg: Message) -> Message | None:
        if msg.type != MessageType.REQUEST:
            return None
        return msg.reply(payload={"echo": msg.payload.get("text", "")}, sender=self.unit_id)


class RouterUnit(WorkingUnit):
    async def handle(self, msg: Message) -> Message | None:
        if msg.type != MessageType.REQUEST:
            return None
        forwarded = await self.request(receiver="echo", payload={"text": msg.payload["text"].upper()})
        return msg.reply(payload={"final": forwarded.payload["echo"]}, sender=self.unit_id)


@pytest.mark.asyncio
async def test_request_response_via_bus():
    bus = AsyncMessageBus()
    echo = EchoUnit(UnitConfig(unit_id="echo", capabilities=["echo"]), bus)
    router = RouterUnit(UnitConfig(unit_id="router", capabilities=["route"]), bus)
    await echo.start()
    await router.start()
    try:
        msg = Message(sender="test", receiver="router", payload={"text": "hello"})
        resp = await bus.request(msg, timeout=5.0)
        assert resp.payload == {"final": "HELLO"}
    finally:
        await echo.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_capability_routing():
    bus = AsyncMessageBus()
    echo = EchoUnit(UnitConfig(unit_id="echo", capabilities=["greet"]), bus)
    await echo.start()
    try:
        msg = Message(sender="test", capability="greet", payload={"text": "hi"})
        resp = await bus.request(msg, timeout=5.0)
        assert resp.payload == {"echo": "hi"}
    finally:
        await echo.stop()


@pytest.mark.asyncio
async def test_unknown_target_does_not_hang():
    bus = AsyncMessageBus()
    msg = Message(sender="test", receiver="ghost", payload={})
    with pytest.raises(asyncio.TimeoutError):
        await bus.request(msg, timeout=0.2)
