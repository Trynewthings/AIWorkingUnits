"""Run the WikiLinter against the local wiki and print a report.

Usage:
  python scripts/lint.py                          # lint ./wiki, report only
  python scripts/lint.py --repair                 # lint + ask Maintainer for fix suggestions (no writes)
  python scripts/lint.py --repair --apply         # lint + apply fixes via Maintainer
  python scripts/lint.py --repair --fix-links     # also opt-in to broken_link de-linkification
  python scripts/lint.py path/to/wiki [flags]     # lint a different wiki dir

Exit code: 0 if no errors remain, 1 if any error-severity issue is found.
Warnings and info-level findings are reported but do not affect exit code.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from aiworkingunits import AsyncMessageBus, Message, load_env

REPO = Path(__file__).resolve().parent.parent
load_env(REPO)

from aiworkingunits.units.wiki_linter import make_linter  # noqa: E402 — env first
from aiworkingunits.units.wiki_maintainer import make_maintainer  # noqa: E402


_SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lint (and optionally repair) the wiki.")
    p.add_argument("wiki_dir", nargs="?", default=str(REPO / "wiki"), help="path to wiki directory")
    p.add_argument("--repair", action="store_true", help="dispatch found issues to Maintainer for repair")
    p.add_argument("--apply", action="store_true", help="with --repair: write fixes to disk (default: suggest only)")
    p.add_argument(
        "--fix-links",
        action="store_true",
        help="with --repair: also opt-in to broken_link de-linkification (default off — destructive)",
    )
    return p.parse_args(argv)


def _print_issues(issues: list[dict], summary: dict) -> None:
    issues = sorted(issues, key=lambda i: (_SEVERITY_RANK.get(i["severity"], 99), i["type"], i["path"]))
    for i in issues:
        tag = f"[{i['severity'].upper():7}] {i['type']:18}"
        detail = f"  ({i['detail']})" if i.get("detail") else ""
        print(f"{tag} {i['path']}: {i['message']}{detail}")
    print()
    print("=== Lint summary ===")
    print(f"Total issues: {summary.get('total', 0)}")
    print(f"By severity:  {summary.get('by_severity', {})}")
    print(f"By type:      {summary.get('by_type', {})}")


def _print_repair(repair: dict) -> None:
    print()
    print(f"=== Repair ({repair.get('mode', '?')}) ===")
    summary = repair.get("summary", {})
    print(f"Considered: {summary.get('considered', 0)}")
    print(f"Proposed:   {summary.get('proposed', 0)}")
    print(f"Fixed:      {summary.get('fixed', 0)}")
    print(f"Skipped:    {summary.get('skipped', 0)}")
    print(f"Errors:     {summary.get('errors', 0)}")
    for prop in repair.get("proposals", []):
        if not prop.get("would_change"):
            continue
        marker = "APPLIED" if prop.get("applied") else "PROPOSED"
        print(f"  [{marker}] {prop['path']}")
        for action in prop.get("actions", []):
            print(f"      - {action}")


async def main(args: argparse.Namespace) -> int:
    bus = AsyncMessageBus()
    maintainer = None
    repair_capability = None
    if args.repair:
        maintainer = make_maintainer(
            bus,
            wiki_dir=args.wiki_dir,
            schema_path=REPO / "schemas" / "book_wiki.md",
        )
        await maintainer.start()
        repair_capability = "wiki.repair"

    repair_policies: dict[str, str] = {}
    if args.fix_links:
        repair_policies["broken_link"] = "fix"

    linter = make_linter(
        bus,
        wiki_dir=args.wiki_dir,
        repair_capability=repair_capability,
        repair_mode="apply" if args.apply else "suggest",
        repair_policies=repair_policies,
    )
    await linter.start()
    try:
        msg = Message(
            sender="cli",
            receiver=linter.unit_id,
            capability="wiki.lint",
            payload={},
        )
        resp = await bus.request(msg, timeout=60.0)
        _print_issues(resp.payload.get("issues", []), resp.payload.get("summary", {}))

        repair = resp.payload.get("repair")
        if repair:
            _print_repair(repair)
        repair_error = resp.payload.get("repair_error")
        if repair_error:
            print(f"\n!! repair dispatch failed: {repair_error}", file=sys.stderr)

        errors = resp.payload.get("summary", {}).get("by_severity", {}).get("error", 0)
        return 1 if errors else 0
    finally:
        await linter.stop()
        if maintainer is not None:
            await maintainer.stop()


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    sys.exit(asyncio.run(main(args)))
