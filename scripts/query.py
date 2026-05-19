"""Ask the wiki a question. Also doubles as a smoke test for LangSmith tracing.

Usage:
  python scripts/query.py "your question here"
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from aiworkingunits import AsyncMessageBus, Message, load_env

REPO = Path(__file__).resolve().parent.parent
load_env(REPO)

from aiworkingunits.units.wiki_maintainer import make_maintainer  # noqa: E402 — env first


async def main(question: str) -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set; copy .env.example to .env")

    bus = AsyncMessageBus()
    maintainer = make_maintainer(
        bus,
        wiki_dir=REPO / "wiki",
        schema_path=REPO / "schemas" / "book_wiki.md",
    )
    await maintainer.start()
    try:
        msg = Message(
            sender="cli",
            receiver=maintainer.unit_id,
            capability="wiki.query",
            payload={"question": question},
        )
        resp = await bus.request(msg, timeout=120.0)
        print("\n=== Answer ===")
        print(resp.payload.get("answer", ""))
        if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
            project = os.getenv("LANGSMITH_PROJECT", "default")
            print(f"\nLangSmith project: {project}  (trace_id={msg.trace_id})")
    finally:
        await maintainer.stop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/query.py 'your question'")
    asyncio.run(main(sys.argv[1]))
