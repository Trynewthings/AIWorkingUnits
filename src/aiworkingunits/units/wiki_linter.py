from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from aiworkingunits.messages import Message, MessageType
from aiworkingunits.unit import UnitConfig, WorkingUnit

logger = logging.getLogger(__name__)


Severity = Literal["error", "warning", "info"]
IssueType = Literal[
    "broken_link",
    "orphan_page",
    "placeholder_text",
    "stub_page",
    "missing_h1",
]


class LintIssue(BaseModel):
    type: IssueType
    path: str
    severity: Severity
    message: str
    detail: str = ""


class WikiLinterConfig(UnitConfig):
    wiki_dir: Path = Path("wiki")
    # Pages smaller than this (in chars, excluding the H1 line) are flagged as stubs.
    stub_min_chars: int = 120
    # Pages without inbound links are flagged as orphans, except those matching
    # these path prefixes (sources/ pages are reachable via the log, not via
    # cross-page links, so they don't need to be orphan-checked).
    orphan_exempt_prefixes: tuple[str, ...] = ("sources/",)
    # When set, after linting the Linter dispatches a request to this capability
    # via the bus with the issue list, then merges the response into its reply.
    # Default None = lint-and-report only (the linter never writes; the writer
    # unit on the other end of repair_capability does).
    repair_capability: str | None = None
    # "suggest" returns proposed changes without writing; "apply" writes them.
    # Per-request payload can override either field.
    repair_mode: Literal["suggest", "apply"] = "suggest"
    # Per-issue-type repair policy; merged on top of the writer's own defaults.
    repair_policies: dict[str, str] = {}


# Red-flag placeholder patterns. Case-insensitive. These exact phrases are
# what burned us once already: an LLM tasked with updating a page emitted a
# stub like "(existing relationship content)" instead of preserving the prior
# text, and the apply step wrote the stub to disk.
_PLACEHOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\(existing\s+[^)]*\)", re.IGNORECASE),
    re.compile(r"\(unchanged[^)]*\)", re.IGNORECASE),
    re.compile(r"\(see\s+previous[^)]*\)", re.IGNORECASE),
    re.compile(r"\(content\s+(?:from|here|omitted)[^)]*\)", re.IGNORECASE),
    re.compile(r"\[\[continue[^\]]*\]\]", re.IGNORECASE),
    re.compile(r"^\s*(TODO|TBD|FIXME|XXX)\b", re.IGNORECASE | re.MULTILINE),
)

# Match markdown links of the form [text](target). Only relative .md targets
# are linted — external URLs and anchors-only links are ignored.
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s#]+(?:#[^)\s]+)?)\)")


class WikiLinter(WorkingUnit):
    """Scans a wiki directory and reports structural quality issues.

    Capability "wiki.lint":
      payload: {} (or {"wiki_dir": "..."} to override)
      reply: {
        "issues": [{type, path, severity, message, detail}],
        "summary": {"total", "by_type", "by_severity"},
      }

    LLM-Wiki-native: no embeddings, no retrieval. Just reads every page and
    applies deterministic checks that compounding-artifact wikis are prone to
    (broken cross-links, placeholder regressions, orphan pages, stubs). These
    checks have no RAG equivalent — they exist precisely because the wiki is
    a structured artifact rather than a chunk soup.
    """

    config_cls = WikiLinterConfig
    config: WikiLinterConfig

    async def handle(self, msg: Message) -> Message | None:
        if msg.type != MessageType.REQUEST:
            return None
        if (msg.capability or "") != "wiki.lint":
            return None
        wiki_dir = Path(msg.payload.get("wiki_dir", self.config.wiki_dir))
        issues = await asyncio.to_thread(self._lint, wiki_dir)
        issue_dicts = [i.model_dump() for i in issues]
        result: dict[str, Any] = {
            "issues": issue_dicts,
            "summary": self._summarize(issues),
        }

        repair_cap = msg.payload.get("repair_capability", self.config.repair_capability)
        if repair_cap and issue_dicts:
            repair_mode = msg.payload.get("repair_mode", self.config.repair_mode)
            repair_policies = {**self.config.repair_policies, **(msg.payload.get("repair_policies") or {})}
            try:
                repair_resp = await self.request(
                    capability=repair_cap,
                    payload={
                        "issues": issue_dicts,
                        "mode": repair_mode,
                        "policies": repair_policies,
                    },
                )
                result["repair"] = repair_resp.payload
            except Exception as e:
                logger.exception("repair dispatch to %s failed", repair_cap)
                result["repair_error"] = {"error": str(e), "error_type": type(e).__name__}

        return msg.reply(payload=result, sender=self.unit_id)

    def _lint(self, wiki_dir: Path) -> list[LintIssue]:
        if not wiki_dir.exists():
            return [
                LintIssue(
                    type="stub_page",
                    path=str(wiki_dir),
                    severity="error",
                    message="wiki directory does not exist",
                )
            ]
        pages = self._collect_pages(wiki_dir)
        issues: list[LintIssue] = []
        issues.extend(self._check_missing_h1(pages))
        issues.extend(self._check_stub_pages(pages))
        issues.extend(self._check_placeholders(pages))
        broken, inbound = self._check_links(pages, wiki_dir)
        issues.extend(broken)
        issues.extend(self._check_orphans(pages, inbound, wiki_dir))
        return issues

    def _collect_pages(self, wiki_dir: Path) -> dict[str, str]:
        """Return {relative-posix-path: body}. Skips index.md, log.md, and .debug/."""
        pages: dict[str, str] = {}
        for p in sorted(wiki_dir.rglob("*.md")):
            rel = p.relative_to(wiki_dir).as_posix()
            if rel in {"index.md", "log.md"} or rel.startswith(".debug/"):
                continue
            pages[rel] = p.read_text(encoding="utf-8")
        return pages

    def _check_missing_h1(self, pages: dict[str, str]) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for rel, body in pages.items():
            first_heading = next(
                (ln.strip() for ln in body.splitlines() if ln.strip().startswith("#")),
                None,
            )
            if first_heading is None or not first_heading.startswith("# "):
                issues.append(
                    LintIssue(
                        type="missing_h1",
                        path=rel,
                        severity="warning",
                        message="page has no top-level # heading",
                    )
                )
        return issues

    def _check_stub_pages(self, pages: dict[str, str]) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for rel, body in pages.items():
            body_no_h1 = re.sub(r"^#\s+.+\n", "", body, count=1).strip()
            if len(body_no_h1) < self.config.stub_min_chars:
                issues.append(
                    LintIssue(
                        type="stub_page",
                        path=rel,
                        severity="warning",
                        message=f"page body is shorter than {self.config.stub_min_chars} chars",
                        detail=f"body length: {len(body_no_h1)} chars",
                    )
                )
        return issues

    def _check_placeholders(self, pages: dict[str, str]) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for rel, body in pages.items():
            for pat in _PLACEHOLDER_PATTERNS:
                m = pat.search(body)
                if m:
                    issues.append(
                        LintIssue(
                            type="placeholder_text",
                            path=rel,
                            severity="error",
                            message="page contains placeholder/stub text — real content may have been overwritten",
                            detail=f"matched: {m.group(0)!r}",
                        )
                    )
                    break
        return issues

    def _check_links(
        self, pages: dict[str, str], wiki_dir: Path
    ) -> tuple[list[LintIssue], dict[str, set[str]]]:
        """Return (broken-link issues, inbound-link map).

        Inbound map: {target-rel-path: {source-rel-path, ...}} used by orphan
        check so we only walk every page once.
        """
        issues: list[LintIssue] = []
        inbound: dict[str, set[str]] = {rel: set() for rel in pages}
        for rel, body in pages.items():
            page_path = wiki_dir / rel
            for match in _LINK_RE.finditer(body):
                target = match.group(2).split("#", 1)[0]
                if not target or self._is_external(target):
                    continue
                if not target.endswith(".md"):
                    continue
                try:
                    resolved = (page_path.parent / target).resolve()
                    rel_target = resolved.relative_to(wiki_dir.resolve()).as_posix()
                except (ValueError, OSError):
                    issues.append(
                        LintIssue(
                            type="broken_link",
                            path=rel,
                            severity="error",
                            message="link escapes wiki_dir or is malformed",
                            detail=f"link text: {match.group(1)!r}, target: {target!r}",
                        )
                    )
                    continue
                if rel_target in pages:
                    inbound[rel_target].add(rel)
                else:
                    issues.append(
                        LintIssue(
                            type="broken_link",
                            path=rel,
                            severity="error",
                            message="link points to a non-existent page",
                            detail=f"link text: {match.group(1)!r}, target: {rel_target}",
                        )
                    )
        return issues, inbound

    def _check_orphans(
        self, pages: dict[str, str], inbound: dict[str, set[str]], wiki_dir: Path
    ) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for rel in pages:
            if rel == "overview.md":
                continue
            if any(rel.startswith(prefix) for prefix in self.config.orphan_exempt_prefixes):
                continue
            if not inbound.get(rel):
                issues.append(
                    LintIssue(
                        type="orphan_page",
                        path=rel,
                        severity="info",
                        message="page is not linked from any other page",
                    )
                )
        return issues

    @staticmethod
    def _is_external(target: str) -> bool:
        return target.startswith(("http://", "https://", "mailto:", "ftp://"))

    @staticmethod
    def _summarize(issues: list[LintIssue]) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for i in issues:
            by_type[i.type] = by_type.get(i.type, 0) + 1
            by_severity[i.severity] = by_severity.get(i.severity, 0) + 1
        return {"total": len(issues), "by_type": by_type, "by_severity": by_severity}


def make_linter(
    bus: Any,
    *,
    unit_id: str = "wiki_linter",
    wiki_dir: Path | str = "wiki",
    stub_min_chars: int = 120,
    repair_capability: str | None = None,
    repair_mode: Literal["suggest", "apply"] = "suggest",
    repair_policies: dict[str, str] | None = None,
) -> WikiLinter:
    cfg = WikiLinterConfig(
        unit_id=unit_id,
        capabilities=["wiki.lint"],
        wiki_dir=Path(wiki_dir),
        stub_min_chars=stub_min_chars,
        repair_capability=repair_capability,
        repair_mode=repair_mode,
        repair_policies=repair_policies or {},
    )
    return WikiLinter(cfg, bus)
