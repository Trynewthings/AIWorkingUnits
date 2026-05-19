from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from aiworkingunits.messages import Message, MessageType
from aiworkingunits.unit import UnitConfig, WorkingUnit

logger = logging.getLogger(__name__)


class SourceFetcherConfig(UnitConfig):
    raw_dir: Path = Path("raw")


class SourceFetcher(WorkingUnit):
    """Reads sources from disk and returns their markdown content.

    Capabilities:
      - "source.fetch": payload {"path": "<relative or absolute path>"}
        returns {"title": str, "content": str, "format": "markdown", "source_path": str}

    Supported formats: .md, .markdown, .pdf (via pymupdf4llm)
    """

    config_cls = SourceFetcherConfig
    config: SourceFetcherConfig

    async def handle(self, msg: Message) -> Message | None:
        if msg.type != MessageType.REQUEST:
            return None
        cap = msg.capability or ""
        if cap != "source.fetch" and msg.payload.get("op") != "fetch":
            return None

        path_str = msg.payload.get("path")
        if not path_str:
            raise ValueError("payload.path is required")

        path = Path(path_str)
        if not path.is_absolute():
            path = (self.config.raw_dir / path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"source not found: {path}")

        content, fmt = await asyncio.to_thread(self._load, path)
        title = self._derive_title(path, content)

        return msg.reply(
            payload={
                "title": title,
                "content": content,
                "format": fmt,
                "source_path": str(path),
            },
            sender=self.unit_id,
        )

    def _load(self, path: Path) -> tuple[str, str]:
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return path.read_text(encoding="utf-8"), "markdown"
        if suffix == ".pdf":
            return self._pdf_to_markdown(path), "markdown"
        raise ValueError(f"unsupported source format: {suffix}")

    def _pdf_to_markdown(self, path: Path) -> str:
        import pymupdf4llm

        logger.info("converting pdf -> markdown: %s", path)
        return pymupdf4llm.to_markdown(str(path))

    @staticmethod
    def _derive_title(path: Path, content: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return path.stem


def make_fetcher(bus: Any, *, unit_id: str = "source_fetcher", raw_dir: Path | str = "raw") -> SourceFetcher:
    cfg = SourceFetcherConfig(
        unit_id=unit_id,
        capabilities=["source.fetch"],
        raw_dir=Path(raw_dir),
    )
    return SourceFetcher(cfg, bus)
