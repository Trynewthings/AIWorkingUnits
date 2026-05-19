from __future__ import annotations

from pathlib import Path

import pytest

from aiworkingunits import AsyncMessageBus, Message
from aiworkingunits.units.source_splitter import SourceSplitter, SourceSplitterConfig, make_splitter


@pytest.mark.asyncio
async def test_split_markdown_with_h1(tmp_path: Path) -> None:
    src = tmp_path / "book.md"
    src.write_text(
        "# Chapter One\n\n"
        + "Once upon a time, there was a quiet village.\n" * 30
        + "\n# Chapter Two\n\n"
        + "Then strange events began to unfold across the country.\n" * 30
        + "\n# Chapter Three\n\n"
        + "By the end, everything had changed.\n" * 30,
        encoding="utf-8",
    )

    bus = AsyncMessageBus()
    unit = make_splitter(bus, raw_dir=tmp_path, target_chapter_chars=0)
    await unit.start()
    try:
        resp = await bus.request(
            Message(
                sender="t",
                receiver=unit.unit_id,
                capability="source.split",
                payload={"path": "book.md"},
            ),
            timeout=5.0,
        )
    finally:
        await unit.stop()

    chapters = resp.payload["chapters"]
    assert resp.payload["strategy"] == "markdown_heading"
    assert [c["title"] for c in chapters] == ["Chapter One", "Chapter Two", "Chapter Three"]
    for c in chapters:
        assert c["chapter_index"] in {0, 1, 2}
        assert c["page_range"] is None
        assert len(c["content"]) >= 500


@pytest.mark.asyncio
async def test_split_markdown_falls_back_to_single_when_no_headings(tmp_path: Path) -> None:
    src = tmp_path / "flat.md"
    src.write_text("a paragraph with no headings.\n" * 50, encoding="utf-8")

    bus = AsyncMessageBus()
    unit = make_splitter(bus, raw_dir=tmp_path, target_chapter_chars=0)
    await unit.start()
    try:
        resp = await bus.request(
            Message(
                sender="t",
                receiver=unit.unit_id,
                capability="source.split",
                payload={"path": "flat.md"},
            ),
            timeout=5.0,
        )
    finally:
        await unit.stop()

    chapters = resp.payload["chapters"]
    assert resp.payload["strategy"] == "single"
    assert len(chapters) == 1
    assert chapters[0]["title"] == "flat"


@pytest.mark.asyncio
async def test_split_drops_tiny_sections(tmp_path: Path) -> None:
    src = tmp_path / "tiny.md"
    src.write_text(
        "# Big Chapter\n\n"
        + "Substantial content.\n" * 60
        + "\n# Tiny\n\nshort\n"
        + "\n# Another Big\n\n"
        + "More substantial content.\n" * 60,
        encoding="utf-8",
    )

    bus = AsyncMessageBus()
    unit = SourceSplitter(
        SourceSplitterConfig(
            unit_id="s",
            capabilities=["source.split"],
            raw_dir=tmp_path,
            min_chapter_chars=500,
            target_chapter_chars=0,
        ),
        bus,
    )
    await unit.start()
    try:
        resp = await bus.request(
            Message(sender="t", receiver="s", capability="source.split", payload={"path": "tiny.md"}),
            timeout=5.0,
        )
    finally:
        await unit.stop()

    titles = [c["title"] for c in resp.payload["chapters"]]
    assert titles == ["Big Chapter", "Another Big"]
