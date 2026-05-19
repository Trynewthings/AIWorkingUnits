"""Split a source into chapters and ingest each into the wiki.

Usage:
  python scripts/ingest.py raw/<file.pdf|file.md>
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from aiworkingunits import AsyncMessageBus, Message, load_env

REPO = Path(__file__).resolve().parent.parent
load_env(REPO)

from aiworkingunits.units.source_splitter import make_splitter  # noqa: E402 - env first
from aiworkingunits.units.wiki_maintainer import make_maintainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("ingest")


def _pick_source(arg: str | None) -> str:
    if arg:
        return arg
    raw = REPO / "raw"
    for p in sorted(raw.iterdir()):
        if p.suffix.lower() in {".md", ".markdown", ".pdf"} and not p.name.startswith("."):
            return p.name
    raise SystemExit("no source in raw/; pass a path or drop a file in raw/")


async def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY not set; copy .env.example to .env")

    source_arg = sys.argv[1] if len(sys.argv) > 1 else None
    source_name = _pick_source(source_arg)
    logger.info("source: %s", source_name)

    bus = AsyncMessageBus()
    splitter = make_splitter(bus, raw_dir=REPO / "raw")
    maintainer = make_maintainer(
        bus,
        wiki_dir=REPO / "wiki",
        schema_path=REPO / "schemas" / "book_wiki.md",
    )
    await splitter.start()
    await maintainer.start()

    try:
        logger.info("step 1: split via %s", splitter.unit_id)
        split_msg = Message(
            sender="cli",
            receiver=splitter.unit_id,
            capability="source.split",
            payload={"path": source_name},
        )
        split_resp = await bus.request(split_msg, timeout=600.0)
        chapters = split_resp.payload.get("chapters", [])
        strategy = split_resp.payload.get("strategy", "?")
        logger.info("got %d chapters (strategy=%s)", len(chapters), strategy)

        if not chapters:
            raise SystemExit("splitter returned no chapters")

        for idx, ch in enumerate(chapters):
            title = ch["title"]
            content_len = len(ch["content"])
            logger.info("step 2.%d/%d ingest: %s (%d chars)", idx + 1, len(chapters), title, content_len)
            ingest_msg = Message(
                sender="cli",
                receiver=maintainer.unit_id,
                capability="wiki.ingest",
                payload={
                    "source_path": ch["source_path"],
                    "title": title,
                    "content": ch["content"],
                },
            )
            ingest_resp = await bus.request(ingest_msg, timeout=900.0)
            applied = ingest_resp.payload.get("applied_pages", [])
            print(f"  + chapter {idx + 1}: {len(applied)} pages -> {', '.join(applied[:5])}{'...' if len(applied) > 5 else ''}")

        print(f"\n=== Done. Ingested {len(chapters)} chapters. ===")
    finally:
        await splitter.stop()
        await maintainer.stop()


if __name__ == "__main__":
    asyncio.run(main())
