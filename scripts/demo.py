"""Two-unit demo: SourceFetcher reads a file from raw/, WikiMaintainer ingests it.

Usage:
  python scripts/demo.py [path-relative-to-raw]

If no path is given, the first .md or .pdf in raw/ is used.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from aiworkingunits import AsyncMessageBus
from aiworkingunits.units.source_fetcher import make_fetcher
from aiworkingunits.units.wiki_maintainer import make_maintainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("demo")

REPO = Path(__file__).resolve().parent.parent


def _pick_source(arg: str | None) -> str:
    if arg:
        return arg
    raw = REPO / "raw"
    for p in sorted(raw.iterdir()):
        if p.suffix.lower() in {".md", ".markdown", ".pdf"} and not p.name.startswith("."):
            return p.name
    raise SystemExit("no source found in raw/; drop a .md or .pdf in there")


async def main() -> None:
    load_dotenv(REPO / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; copy .env.example to .env and fill it in")

    source_arg = sys.argv[1] if len(sys.argv) > 1 else None
    source_name = _pick_source(source_arg)
    logger.info("source: %s", source_name)

    bus = AsyncMessageBus()
    fetcher = make_fetcher(bus, raw_dir=REPO / "raw")
    maintainer = make_maintainer(
        bus,
        wiki_dir=REPO / "wiki",
        schema_path=REPO / "schemas" / "book_wiki.md",
    )
    await fetcher.start()
    await maintainer.start()

    try:
        logger.info("step 1: requesting source.fetch from %s", fetcher.unit_id)
        fetched = await maintainer.request(
            receiver=fetcher.unit_id,
            payload={"path": source_name},
        )
        logger.info("fetched title=%r bytes=%d", fetched.payload.get("title"), len(fetched.payload.get("content", "")))

        logger.info("step 2: requesting wiki.ingest from %s", maintainer.unit_id)
        ingest_msg = await fetcher.request(
            receiver=maintainer.unit_id,
            capability="wiki.ingest",
            payload={
                "source_path": fetched.payload["source_path"],
                "title": fetched.payload["title"],
                "content": fetched.payload["content"],
            },
            timeout=300.0,
        )
        applied = ingest_msg.payload.get("applied_pages", [])
        summary = ingest_msg.payload.get("summary", "")
        print("\n=== Ingest result ===")
        print(f"summary: {summary}")
        print(f"pages touched ({len(applied)}):")
        for p in applied:
            print(f"  - {p}")
    finally:
        await fetcher.stop()
        await maintainer.stop()


if __name__ == "__main__":
    asyncio.run(main())
