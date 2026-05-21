"""Unified command-line entry point for AI Working Units.

Run `workunits --help` after `pip install -e .`, or invoke as
`python -m aiworkingunits.cli` if no entry point is installed.

Subcommands:
  status                       — print wiki stats (no LLM needed)
  ingest <path>                — split a source and ingest each chunk
  query "<question>"           — ask the wiki; cites pages used
  lint [--repair] [--apply]    — scan wiki for structural issues
  log  [--last N]              — tail the ingest log

The CLI is a thin coordinator over the bus + units. All actual work
happens inside `WorkingUnit` instances; the CLI just orchestrates messages.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from aiworkingunits import AsyncMessageBus, Message, load_env

REPO = Path(__file__).resolve().parents[2]


# ----- terminal styling (no extra deps) ----------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t: str) -> str:
    return _c(t, "1")


def _dim(t: str) -> str:
    return _c(t, "2")


def _red(t: str) -> str:
    return _c(t, "31")


def _green(t: str) -> str:
    return _c(t, "32")


def _yellow(t: str) -> str:
    return _c(t, "33")


def _cyan(t: str) -> str:
    return _c(t, "36")


_SEVERITY_COLORS = {"error": _red, "warning": _yellow, "info": _dim}


# ----- helpers -----------------------------------------------------------


def _resolve_wiki_dir(args: argparse.Namespace) -> Path:
    return Path(args.wiki_dir).resolve() if args.wiki_dir else REPO / "wiki"


def _resolve_raw_dir(args: argparse.Namespace) -> Path:
    return Path(args.raw_dir).resolve() if args.raw_dir else REPO / "raw"


def _resolve_schema_path(args: argparse.Namespace) -> Path:
    return Path(args.schema).resolve() if args.schema else REPO / "schemas" / "book_wiki.md"


def _require_env(*keys: str) -> None:
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            _red(f"missing env: {', '.join(missing)}. ")
            + _dim("Create .env from .env.example or export the keys.")
        )


# ----- status ------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    wiki_dir = _resolve_wiki_dir(args)
    if not wiki_dir.exists():
        print(_red(f"wiki dir not found: {wiki_dir}"))
        return 1
    pages = sorted(p for p in wiki_dir.rglob("*.md")
                   if not p.relative_to(wiki_dir).as_posix().startswith(".debug/"))
    total_chars = sum(p.stat().st_size for p in pages)
    by_cat: dict[str, int] = {}
    for p in pages:
        rel = p.relative_to(wiki_dir).as_posix()
        cat = rel.split("/", 1)[0] if "/" in rel else "root"
        by_cat[cat] = by_cat.get(cat, 0) + 1

    print(_bold("Wiki status"))
    print(f"  path:        {wiki_dir}")
    print(f"  pages:       {len(pages)}")
    print(f"  total size:  {total_chars:,} bytes (~{total_chars // 1024} KB)")
    print(f"  by category:")
    for cat in sorted(by_cat):
        print(f"    {cat:12} {by_cat[cat]}")

    log_path = wiki_dir / "log.md"
    if log_path.exists():
        entries = [ln for ln in log_path.read_text(encoding="utf-8").splitlines()
                   if ln.startswith("## [")]
        print(f"  ingests:     {len(entries)}")
        if entries:
            print(f"  last ingest: {_dim(entries[-1].lstrip('# ').strip())}")
    return 0


# ----- log ---------------------------------------------------------------


def cmd_log(args: argparse.Namespace) -> int:
    wiki_dir = _resolve_wiki_dir(args)
    log_path = wiki_dir / "log.md"
    if not log_path.exists():
        print(_dim("(no log yet)"))
        return 0
    text = log_path.read_text(encoding="utf-8")
    blocks: list[list[str]] = []
    current: list[str] = []
    for ln in text.splitlines():
        if ln.startswith("## ["):
            if current:
                blocks.append(current)
            current = [ln]
        elif current:
            current.append(ln)
    if current:
        blocks.append(current)
    selected = blocks[-args.last:] if args.last > 0 else blocks
    for block in selected:
        print(_cyan(block[0]))
        for ln in block[1:]:
            print(ln)
        print()
    return 0


# ----- ingest ------------------------------------------------------------


async def _do_ingest(args: argparse.Namespace) -> int:
    from aiworkingunits.units.source_splitter import make_splitter
    from aiworkingunits.units.wiki_maintainer import make_maintainer

    _require_env("DEEPSEEK_API_KEY")
    wiki_dir = _resolve_wiki_dir(args)
    raw_dir = _resolve_raw_dir(args)
    schema = _resolve_schema_path(args)

    bus = AsyncMessageBus()
    splitter = make_splitter(bus, raw_dir=raw_dir)
    maintainer = make_maintainer(bus, wiki_dir=wiki_dir, schema_path=schema)
    await splitter.start()
    await maintainer.start()
    try:
        print(_bold(f"Step 1: splitting {args.path}"))
        split_resp = await bus.request(
            Message(sender="cli", receiver=splitter.unit_id, capability="source.split",
                    payload={"path": args.path}),
            timeout=600.0,
        )
        chapters = split_resp.payload.get("chapters", [])
        strategy = split_resp.payload.get("strategy", "?")
        if not chapters:
            print(_red("splitter returned no chapters"))
            return 1
        print(f"  got {_green(str(len(chapters)))} chunks (strategy={strategy})")

        if args.dry_run:
            for i, ch in enumerate(chapters):
                print(f"  {i + 1:2}. {ch['title']!r} ({len(ch['content']):,} chars)")
            print(_dim("\n(dry-run: skipping ingest)"))
            return 0

        print(_bold(f"\nStep 2: ingesting {len(chapters)} chunks"))
        for i, ch in enumerate(chapters):
            print(f"  [{i + 1}/{len(chapters)}] {ch['title']} ({len(ch['content']):,} chars)")
            ingest_resp = await bus.request(
                Message(sender="cli", receiver=maintainer.unit_id, capability="wiki.ingest",
                        payload={"source_path": ch["source_path"], "title": ch["title"],
                                 "content": ch["content"]}),
                timeout=900.0,
            )
            applied = ingest_resp.payload.get("applied_pages", [])
            preview = ", ".join(applied[:4]) + ("..." if len(applied) > 4 else "")
            print(f"        {_green('+')} {len(applied)} pages: {_dim(preview)}")

        print(_bold(_green(f"\nDone. Ingested {len(chapters)} chunks into {wiki_dir.name}/.")))
        return 0
    finally:
        await splitter.stop()
        await maintainer.stop()


def cmd_ingest(args: argparse.Namespace) -> int:
    return asyncio.run(_do_ingest(args))


# ----- query -------------------------------------------------------------


async def _do_query(args: argparse.Namespace) -> int:
    from aiworkingunits.units.wiki_maintainer import make_maintainer

    _require_env("DEEPSEEK_API_KEY")
    wiki_dir = _resolve_wiki_dir(args)
    schema = _resolve_schema_path(args)

    bus = AsyncMessageBus()
    maintainer = make_maintainer(bus, wiki_dir=wiki_dir, schema_path=schema)
    await maintainer.start()
    try:
        resp = await bus.request(
            Message(sender="cli", receiver=maintainer.unit_id, capability="wiki.query",
                    payload={"question": args.question}),
            timeout=120.0,
        )
        cited = resp.payload.get("cited_pages", [])
        reasoning = resp.payload.get("reasoning", "")
        answer = resp.payload.get("answer", "")

        if cited:
            print(_bold("Sources:"))
            for c in cited:
                print(f"  - {_cyan(c)}")
            if reasoning and args.verbose:
                print(_dim(f"  ({reasoning})"))
            print()
        print(_bold("Answer:"))
        print(answer)

        if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
            project = os.environ.get("LANGSMITH_PROJECT", "default")
            print(_dim(f"\n(LangSmith project: {project})"))
        return 0
    finally:
        await maintainer.stop()


def cmd_query(args: argparse.Namespace) -> int:
    return asyncio.run(_do_query(args))


# ----- lint --------------------------------------------------------------


_LINT_SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}


async def _do_lint(args: argparse.Namespace) -> int:
    from aiworkingunits.units.wiki_linter import make_linter
    from aiworkingunits.units.wiki_maintainer import make_maintainer

    wiki_dir = _resolve_wiki_dir(args)

    bus = AsyncMessageBus()
    maintainer = None
    repair_capability = None
    if args.repair:
        maintainer = make_maintainer(
            bus, wiki_dir=wiki_dir, schema_path=_resolve_schema_path(args)
        )
        await maintainer.start()
        repair_capability = "wiki.repair"

    repair_policies: dict[str, str] = {}
    if args.fix_links:
        repair_policies["broken_link"] = "fix"

    linter = make_linter(
        bus, wiki_dir=wiki_dir,
        repair_capability=repair_capability,
        repair_mode="apply" if args.apply else "suggest",
        repair_policies=repair_policies,
    )
    await linter.start()
    try:
        resp = await bus.request(
            Message(sender="cli", receiver=linter.unit_id, capability="wiki.lint", payload={}),
            timeout=60.0,
        )
        issues = sorted(resp.payload.get("issues", []),
                        key=lambda i: (_LINT_SEVERITY_RANK.get(i["severity"], 99), i["type"], i["path"]))
        for i in issues:
            color = _SEVERITY_COLORS.get(i["severity"], lambda t: t)
            tag = color(f"[{i['severity'].upper():7}] {i['type']:18}")
            detail = _dim(f"  ({i['detail']})") if i.get("detail") else ""
            print(f"{tag} {i['path']}: {i['message']}{detail}")

        summary = resp.payload.get("summary", {})
        print()
        print(_bold("Lint summary"))
        print(f"  total:        {summary.get('total', 0)}")
        print(f"  by severity:  {summary.get('by_severity', {})}")
        print(f"  by type:      {summary.get('by_type', {})}")

        repair = resp.payload.get("repair")
        if repair:
            rs = repair.get("summary", {})
            print()
            print(_bold(f"Repair ({repair.get('mode', '?')})"))
            print(f"  considered: {rs.get('considered', 0)}")
            print(f"  proposed:   {rs.get('proposed', 0)}")
            print(f"  fixed:      {_green(str(rs.get('fixed', 0)))}")
            print(f"  skipped:    {rs.get('skipped', 0)}")
            for prop in repair.get("proposals", []):
                if not prop.get("would_change"):
                    continue
                tag = _green("[APPLIED]") if prop.get("applied") else _yellow("[PROPOSED]")
                print(f"  {tag} {prop['path']}")
                for action in prop.get("actions", []):
                    print(f"      - {action}")

        repair_error = resp.payload.get("repair_error")
        if repair_error:
            print(_red(f"\n!! repair dispatch failed: {repair_error}"))

        errors = summary.get("by_severity", {}).get("error", 0)
        return 1 if errors else 0
    finally:
        await linter.stop()
        if maintainer is not None:
            await maintainer.stop()


def cmd_lint(args: argparse.Namespace) -> int:
    return asyncio.run(_do_lint(args))


# ----- argparse setup ----------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="workunits",
        description="AI Working Units — wiki + agents over an async message bus.",
    )
    p.add_argument("--wiki-dir", help="override wiki directory (default: ./wiki)")
    p.add_argument("--raw-dir", help="override raw sources directory (default: ./raw)")
    p.add_argument("--schema", help="override wiki schema (default: ./schemas/book_wiki.md)")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="print wiki stats")

    p_log = sub.add_parser("log", help="show ingest log entries")
    p_log.add_argument("--last", type=int, default=5, help="show last N entries (default 5; 0 = all)")

    p_ing = sub.add_parser("ingest", help="split + ingest a source")
    p_ing.add_argument("path", help="path to source (e.g. raw/<file>.pdf or absolute)")
    p_ing.add_argument("--dry-run", action="store_true", help="only split, don't ingest")

    p_q = sub.add_parser("query", help="ask the wiki")
    p_q.add_argument("question", help="your question (quote it)")

    p_lint = sub.add_parser("lint", help="scan wiki for structural issues")
    p_lint.add_argument("--repair", action="store_true",
                        help="dispatch found issues to Maintainer for repair")
    p_lint.add_argument("--apply", action="store_true",
                        help="with --repair: write fixes to disk (default: suggest only)")
    p_lint.add_argument("--fix-links", action="store_true",
                        help="with --repair: also opt-in to broken_link de-linkification")

    return p


_CMD_TABLE = {
    "status": cmd_status,
    "log": cmd_log,
    "ingest": cmd_ingest,
    "query": cmd_query,
    "lint": cmd_lint,
}


def main(argv: list[str] | None = None) -> int:
    load_env(REPO)
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    handler = _CMD_TABLE.get(args.cmd)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return handler(args)
    except KeyboardInterrupt:
        print(_dim("\ninterrupted"))
        return 130
    except SystemExit:
        raise
    except Exception as e:
        if args.verbose:
            raise
        print(_red(f"error: {type(e).__name__}: {e}"))
        print(_dim("(re-run with -v for full traceback)"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
