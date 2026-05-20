from __future__ import annotations

from pathlib import Path

import pytest

from aiworkingunits import AsyncMessageBus, Message
from aiworkingunits.units.wiki_linter import make_linter
from aiworkingunits.units.wiki_maintainer import make_maintainer


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def _make_wiki(root: Path) -> None:
    """A fixture wiki containing one example of each lint issue plus clean pages."""
    _write(root / "index.md", "# Index\n")
    _write(root / "log.md", "# Log\n")
    _write(
        root / "overview.md",
        "# Overview\n\nLinks to [Alice](entities/alice.md) and [Concept A](concepts/concept-a.md).\n",
    )
    _write(
        root / "entities" / "alice.md",
        "# Alice\n\nA reasonably sized page that links to [Concept A](../concepts/concept-a.md)"
        " and is itself linked from the overview. Padding to clear the stub threshold easily.\n",
    )
    _write(
        root / "concepts" / "concept-a.md",
        "# Concept A\n\nA real concept page linked from both alice.md and overview.md."
        " Body is comfortably above the stub threshold so it stays clean.\n",
    )
    # Bad: placeholder text — exactly the regression we hit.
    _write(
        root / "concepts" / "concept-b.md",
        "# Concept B\n\n## Definition\n(Existing content here)\n\n## Other\nReal text.\n",
    )
    # Bad: broken link to a non-existent page.
    _write(
        root / "concepts" / "broken.md",
        "# Broken\n\nThis links to [a ghost](../entities/ghost.md) that doesn't exist."
        " Lots of padding so this page itself isn't a stub.\n",
    )
    # Bad: very short body.
    _write(root / "concepts" / "stubby.md", "# Stubby\n\nTiny.\n")
    # Bad: missing H1.
    _write(root / "concepts" / "no-h1.md", "Just a paragraph, no heading, plenty of length to clear stub.\n")
    # Bad: orphan page (not linked from anywhere).
    _write(
        root / "entities" / "orphan.md",
        "# Orphan\n\nNobody links to me but I have plenty of content of my own here.\n",
    )
    # Source page — exempt from orphan check even though nothing links to it.
    _write(
        root / "sources" / "src-1.md",
        "# Source: example\n\nA source summary page; orphan-exempt by convention.\n",
    )


@pytest.mark.asyncio
async def test_linter_reports_each_issue_type(tmp_path: Path):
    _make_wiki(tmp_path)
    bus = AsyncMessageBus()
    linter = make_linter(bus, wiki_dir=tmp_path)
    await linter.start()
    try:
        msg = Message(sender="test", capability="wiki.lint", payload={})
        resp = await bus.request(msg, timeout=5.0)
    finally:
        await linter.stop()

    issues = resp.payload["issues"]
    by_type: dict[str, list[dict]] = {}
    for i in issues:
        by_type.setdefault(i["type"], []).append(i)

    paths_for = lambda t: {i["path"] for i in by_type.get(t, [])}

    assert "concepts/concept-b.md" in paths_for("placeholder_text")
    assert "concepts/broken.md" in paths_for("broken_link")
    assert "concepts/stubby.md" in paths_for("stub_page")
    assert "concepts/no-h1.md" in paths_for("missing_h1")
    assert "entities/orphan.md" in paths_for("orphan_page")

    # Source pages are orphan-exempt, overview is exempt, linked pages are not orphans.
    assert "sources/src-1.md" not in paths_for("orphan_page")
    assert "overview.md" not in paths_for("orphan_page")
    assert "entities/alice.md" not in paths_for("orphan_page")
    assert "concepts/concept-a.md" not in paths_for("orphan_page")


@pytest.mark.asyncio
async def test_linter_dispatches_repair_to_maintainer_apply(tmp_path: Path):
    """End-to-end: Linter finds issues, sends them to Maintainer via bus, Maintainer fixes."""
    # missing_h1: no leading # heading, but enough body to not be a stub.
    _write(
        tmp_path / "concepts" / "needs-heading.md",
        "Some body content that is comfortably above the stub threshold and"
        " describes a concept that has no H1 at the top. Padding padding padding.\n",
    )
    # broken_link: a real H1, padded body, one link to a ghost page.
    _write(
        tmp_path / "entities" / "broken.md",
        "# Broken\n\nThis page mentions a [Ghost](../entities/ghost.md) that does"
        " not exist. The page itself is well above the stub threshold so only the"
        " broken_link issue should fire.\n",
    )

    schema = tmp_path.parent / "schema.md"
    schema.write_text("# fake schema\n", encoding="utf-8")

    bus = AsyncMessageBus()
    maintainer = make_maintainer(bus, wiki_dir=tmp_path, schema_path=schema)
    linter = make_linter(
        bus,
        wiki_dir=tmp_path,
        repair_capability="wiki.repair",
        repair_mode="apply",
        repair_policies={"broken_link": "fix"},
    )
    await maintainer.start()
    await linter.start()
    try:
        msg = Message(sender="test", capability="wiki.lint", payload={})
        resp = await bus.request(msg, timeout=10.0)
    finally:
        await linter.stop()
        await maintainer.stop()

    repair = resp.payload.get("repair")
    assert repair is not None, "Linter did not dispatch repair to the Maintainer"
    assert repair["mode"] == "apply"
    assert repair["summary"]["fixed"] >= 2

    # missing_h1 was fixed
    fixed_h1 = (tmp_path / "concepts" / "needs-heading.md").read_text(encoding="utf-8")
    assert fixed_h1.startswith("# Needs Heading")

    # broken_link was de-linkified
    fixed_link = (tmp_path / "entities" / "broken.md").read_text(encoding="utf-8")
    assert "[Ghost]" not in fixed_link
    assert "Ghost" in fixed_link  # the visible text survives

    # Re-running lint should now show no errors of these types
    bus2 = AsyncMessageBus()
    linter2 = make_linter(bus2, wiki_dir=tmp_path)
    await linter2.start()
    try:
        msg = Message(sender="test", capability="wiki.lint", payload={})
        resp2 = await bus2.request(msg, timeout=5.0)
    finally:
        await linter2.stop()
    types_after = {i["type"] for i in resp2.payload["issues"]}
    assert "missing_h1" not in types_after
    assert "broken_link" not in types_after


@pytest.mark.asyncio
async def test_linter_repair_suggest_does_not_write(tmp_path: Path):
    """Suggest mode returns proposals but leaves the filesystem untouched."""
    page = tmp_path / "concepts" / "needs-heading.md"
    _write(page, "Body text plenty long enough to clear the stub threshold without an H1.\n")
    original = page.read_text(encoding="utf-8")

    schema = tmp_path.parent / "schema.md"
    schema.write_text("# fake schema\n", encoding="utf-8")

    bus = AsyncMessageBus()
    maintainer = make_maintainer(bus, wiki_dir=tmp_path, schema_path=schema)
    linter = make_linter(
        bus,
        wiki_dir=tmp_path,
        repair_capability="wiki.repair",
        repair_mode="suggest",
    )
    await maintainer.start()
    await linter.start()
    try:
        msg = Message(sender="test", capability="wiki.lint", payload={})
        resp = await bus.request(msg, timeout=10.0)
    finally:
        await linter.stop()
        await maintainer.stop()

    repair = resp.payload["repair"]
    assert repair["mode"] == "suggest"
    assert repair["summary"]["proposed"] >= 1
    assert repair["summary"]["fixed"] == 0
    # Filesystem untouched.
    assert page.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_linter_repair_skips_broken_link_by_default(tmp_path: Path):
    """Default policy keeps broken_link as skip; user must opt-in via policies."""
    page = tmp_path / "entities" / "broken.md"
    _write(
        page,
        "# Broken\n\nLinks to a [Ghost](../entities/ghost.md) that's missing."
        " Otherwise this page has plenty of body to avoid the stub flag.\n",
    )
    original = page.read_text(encoding="utf-8")

    schema = tmp_path.parent / "schema.md"
    schema.write_text("# fake schema\n", encoding="utf-8")

    bus = AsyncMessageBus()
    maintainer = make_maintainer(bus, wiki_dir=tmp_path, schema_path=schema)
    linter = make_linter(
        bus,
        wiki_dir=tmp_path,
        repair_capability="wiki.repair",
        repair_mode="apply",
        # Note: no broken_link policy override
    )
    await maintainer.start()
    await linter.start()
    try:
        msg = Message(sender="test", capability="wiki.lint", payload={})
        resp = await bus.request(msg, timeout=10.0)
    finally:
        await linter.stop()
        await maintainer.stop()

    repair = resp.payload["repair"]
    # No fixers ran — the only issue was broken_link, and policy=skip.
    assert repair["summary"]["fixed"] == 0
    skipped_types = {s["type"] for s in repair.get("skipped", [])}
    assert "broken_link" in skipped_types
    assert page.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_linter_clean_wiki(tmp_path: Path):
    """A minimal wiki with only well-formed pages should produce no errors."""
    _write(tmp_path / "index.md", "# Index\n")
    _write(tmp_path / "log.md", "# Log\n")
    _write(
        tmp_path / "overview.md",
        "# Overview\n\nThis points at [Alice](entities/alice.md), a real page."
        " Plenty of content here so the body easily clears the stub threshold.\n",
    )
    _write(
        tmp_path / "entities" / "alice.md",
        "# Alice\n\nA real entity page linked from the overview, well above the stub"
        " threshold and with no placeholders or broken links anywhere in sight.\n",
    )

    bus = AsyncMessageBus()
    linter = make_linter(bus, wiki_dir=tmp_path)
    await linter.start()
    try:
        msg = Message(sender="test", capability="wiki.lint", payload={})
        resp = await bus.request(msg, timeout=5.0)
    finally:
        await linter.stop()

    errors = [i for i in resp.payload["issues"] if i["severity"] == "error"]
    assert errors == []
