from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from aiworkingunits.messages import Message, MessageType
from aiworkingunits.unit import UnitConfig, WorkingUnit

logger = logging.getLogger(__name__)


class SourceSplitterConfig(UnitConfig):
    raw_dir: Path = Path("raw")
    min_chapter_chars: int = 500
    # When raw sections are smaller than this, greedy-merge consecutive ones.
    # Books often have many short sub-sections that pymupdf4llm flattens to a
    # single heading level; without packing we end up with 80+ "chapters" of
    # 1-2 KB each, which is the wrong granularity for ingest.
    target_chapter_chars: int = 15000


class SourceSplitter(WorkingUnit):
    """Splits a long source (PDF or markdown) into chapter-sized chunks.

    Capability "source.split":
      payload: {"path": "..."}
      reply: {
        "chapters": [{"title", "content", "source_path", "chapter_index", "page_range"}],
        "source_path": str,
        "strategy": "pdf_toc" | "markdown_heading" | "single",
      }

    Strategy:
      - PDF with bookmarks → use pymupdf TOC (precise page-range chapters)
      - PDF without bookmarks → convert whole doc to markdown, split on H1/H2
      - markdown → split on H1/H2
      - small chapters (<min_chapter_chars) are dropped to skip TOC headers,
        copyright pages, etc.
    """

    config_cls = SourceSplitterConfig
    config: SourceSplitterConfig

    async def handle(self, msg: Message) -> Message | None:
        if msg.type != MessageType.REQUEST:
            return None
        if (msg.capability or "") != "source.split":
            return None

        path_str = msg.payload.get("path")
        if not path_str:
            raise ValueError("payload.path is required")
        path = Path(path_str)
        if not path.is_absolute():
            path = (self.config.raw_dir / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"source not found: {path}")

        chapters, strategy = await asyncio.to_thread(self._split, path)
        return msg.reply(
            payload={"chapters": chapters, "source_path": str(path), "strategy": strategy},
            sender=self.unit_id,
        )

    def _split(self, path: Path) -> tuple[list[dict[str, Any]], str]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._split_pdf(path)
        if suffix in {".md", ".markdown"}:
            return self._split_markdown(path.read_text(encoding="utf-8"), path)
        raise ValueError(f"unsupported format: {suffix}")

    def _split_pdf(self, path: Path) -> tuple[list[dict[str, Any]], str]:
        import pymupdf
        import pymupdf4llm

        doc = pymupdf.open(str(path))
        toc = doc.get_toc()
        top_level = [e for e in toc if e[0] == 1]
        if not top_level:
            logger.info("no PDF TOC; converting whole file and splitting by heading: %s", path.name)
            md = pymupdf4llm.to_markdown(str(path))
            return self._split_markdown(md, path)

        logger.info("splitting PDF by TOC: %d top-level entries", len(top_level))
        chapters: list[dict[str, Any]] = []
        for i, entry in enumerate(top_level):
            _, title, start_page_1idx = entry[0], entry[1], entry[2]
            next_start = top_level[i + 1][2] if i + 1 < len(top_level) else doc.page_count + 1
            page_indices = list(
                range(max(0, start_page_1idx - 1), min(doc.page_count, next_start - 1))
            )
            if not page_indices:
                continue
            md = pymupdf4llm.to_markdown(doc, pages=page_indices)
            if len(md.strip()) < self.config.min_chapter_chars:
                logger.debug("skipping tiny chapter %r (%d chars)", title, len(md))
                continue
            chapters.append(
                {
                    "title": title.strip(),
                    "content": md,
                    "source_path": str(path),
                    "chapter_index": len(chapters),
                    "page_range": [start_page_1idx, next_start - 1],
                }
            )
        chapters = self._pack(chapters)
        return chapters, "pdf_toc"

    def _split_markdown(self, text: str, path: Path) -> tuple[list[dict[str, Any]], str]:
        h1 = self._heading_offsets(text, level=1)
        h2 = self._heading_offsets(text, level=2)
        boundaries = h1 if len(h1) >= 2 else h2
        if len(boundaries) < 2:
            title = self._first_heading(text) or path.stem
            return [
                {
                    "title": title,
                    "content": text,
                    "source_path": str(path),
                    "chapter_index": 0,
                    "page_range": None,
                }
            ], "single"

        chapters: list[dict[str, Any]] = []
        for i, (start, title) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            chunk = text[start:end].strip()
            if len(chunk) < self.config.min_chapter_chars:
                continue
            chapters.append(
                {
                    "title": title,
                    "content": chunk,
                    "source_path": str(path),
                    "chapter_index": len(chapters),
                    "page_range": None,
                }
            )
        chapters = self._pack(chapters)
        return chapters, "markdown_heading"

    def _pack(self, chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Greedy-merge consecutive small chapters until each reaches target size.

        Keeps single chapters that already exceed target as-is (don't split mid-section).
        Merged chapter titles become "First Title -> Last Title".
        """
        target = self.config.target_chapter_chars
        if target <= 0 or not chapters:
            return chapters
        packed: list[dict[str, Any]] = []
        cur: dict[str, Any] | None = None
        for ch in chapters:
            if cur is None:
                cur = dict(ch)
                cur["_first_title"] = ch["title"]
                continue
            if len(cur["content"]) + len(ch["content"]) <= target:
                cur["content"] = cur["content"].rstrip() + "\n\n" + ch["content"].lstrip()
                cur["title"] = f'{cur["_first_title"]} -> {ch["title"]}'
                if ch.get("page_range") and cur.get("page_range"):
                    cur["page_range"] = [cur["page_range"][0], ch["page_range"][1]]
            else:
                cur.pop("_first_title", None)
                packed.append(cur)
                cur = dict(ch)
                cur["_first_title"] = ch["title"]
        if cur is not None:
            cur.pop("_first_title", None)
            packed.append(cur)
        for i, c in enumerate(packed):
            c["chapter_index"] = i
        return packed

    @staticmethod
    def _heading_offsets(text: str, level: int) -> list[tuple[int, str]]:
        prefix = "#" * level + " "
        excluded_prefix = "#" * (level + 1) + " "
        results: list[tuple[int, str]] = []
        pos = 0
        for line in text.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith(prefix) and not stripped.startswith(excluded_prefix):
                title = stripped[len(prefix):].strip().lstrip("#").strip()
                title = title or f"section-{len(results) + 1}"
                results.append((pos, title))
            pos += len(line)
        return results

    @staticmethod
    def _first_heading(text: str) -> str | None:
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("# ") and not s.startswith("## "):
                return s[2:].strip()
        return None


def make_splitter(
    bus: Any,
    *,
    unit_id: str = "source_splitter",
    raw_dir: Path | str = "raw",
    target_chapter_chars: int = 15000,
    min_chapter_chars: int = 500,
) -> SourceSplitter:
    cfg = SourceSplitterConfig(
        unit_id=unit_id,
        capabilities=["source.split"],
        raw_dir=Path(raw_dir),
        target_chapter_chars=target_chapter_chars,
        min_chapter_chars=min_chapter_chars,
    )
    return SourceSplitter(cfg, bus)
