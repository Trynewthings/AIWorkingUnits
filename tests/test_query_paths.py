"""Path-validation unit tests for WikiMaintainer's query graph.

These tests exercise the deterministic IO node (_read_selected_pages) without
ever hitting an LLM. They cover the security-relevant cases: paths that escape
wiki_dir, non-markdown paths, non-existent pages, embedded newlines, etc.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aiworkingunits import AsyncMessageBus
from aiworkingunits.units.wiki_maintainer import make_maintainer


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


@pytest.fixture
def maintainer(tmp_path: Path):
    _write(tmp_path / "index.md", "# Index\n")
    _write(tmp_path / "concepts" / "foo.md", "# Foo\n\nFoo content.\n")
    _write(tmp_path / "entities" / "bar.md", "# Bar\n\nBar content.\n")
    schema = tmp_path.parent / "schema.md"
    schema.write_text("# fake schema\n", encoding="utf-8")
    bus = AsyncMessageBus()
    return make_maintainer(bus, wiki_dir=tmp_path, schema_path=schema)


def test_loads_existing_pages(maintainer):
    out = maintainer._read_selected_pages(["concepts/foo.md", "entities/bar.md"])
    assert set(out.keys()) == {"concepts/foo.md", "entities/bar.md"}
    assert "Foo content" in out["concepts/foo.md"]


def test_drops_non_existent_pages(maintainer):
    out = maintainer._read_selected_pages(["concepts/foo.md", "concepts/ghost.md"])
    assert "concepts/ghost.md" not in out
    assert "concepts/foo.md" in out


def test_rejects_path_traversal(maintainer):
    # An LLM might hallucinate a path like "../../etc/passwd" — must be dropped.
    out = maintainer._read_selected_pages(["../../etc/passwd", "concepts/foo.md"])
    assert all(not k.endswith("passwd") for k in out)
    assert "concepts/foo.md" in out


def test_rejects_non_markdown(maintainer):
    out = maintainer._read_selected_pages(["concepts/foo.txt", "raw/bin.pdf"])
    assert out == {}


def test_rejects_paths_with_newlines(maintainer):
    out = maintainer._read_selected_pages(["concepts/foo.md\nrm -rf /"])
    assert out == {}


def test_strips_leading_slash(maintainer):
    out = maintainer._read_selected_pages(["/concepts/foo.md"])
    assert "concepts/foo.md" in out


def test_empty_input(maintainer):
    assert maintainer._read_selected_pages([]) == {}
