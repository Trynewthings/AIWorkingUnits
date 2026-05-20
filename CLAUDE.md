# Project context for new Claude Code sessions

This file is auto-loaded by Claude Code when it opens this repo. Read it
before doing anything else — it captures hard-won design decisions and
the conceptual framing so a fresh session doesn't redo the same wrong
turns.

---

## What this project is

**AI Working Units** — a small framework where each "unit" is a
configurable AI agent (LangGraph + LangSmith) that talks to other units
over an async message bus. Same code, different `UnitConfig` = a unit
retargeted at a similar task.

The **first concrete application** built on top is the **LLM Wiki**
pattern (idea attribution and rationale in the LLM Wiki section below).
The units in service of that:

| Unit | File | Capabilities |
|---|---|---|
| `SourceSplitter` | `src/aiworkingunits/units/source_splitter.py` | `source.split` — PDF (TOC-aware) or markdown → list of chapter-sized chunks, greedy-packed to `target_chapter_chars` |
| `WikiMaintainer` | `src/aiworkingunits/units/wiki_maintainer.py` | `wiki.ingest`, `wiki.query`, `wiki.repair` (deterministic fixers for `missing_h1`, `broken_link`; suggest/apply modes; per-issue-type policies) |
| `WikiLinter` | `src/aiworkingunits/units/wiki_linter.py` | `wiki.lint` — deterministic structural checks (broken links, placeholder text, stub pages, orphan pages, missing H1). Can dispatch found issues to a `wiki.repair`-capable unit via the bus for closed-loop self-repair. |
| `SourceFetcher` | `src/aiworkingunits/units/source_fetcher.py` | `source.fetch` — read raw file as markdown (PDF via pymupdf4llm); kept around for direct single-file ingest demos |

Bus + ABC live in `src/aiworkingunits/{bus,unit,messages,observability}.py`.

The **wiki schema** (page types, link conventions, ingest rules) is in
`schemas/book_wiki.md`. Retargeting the WikiMaintainer to a different
domain = edit that file, no code change.

---

## CRITICAL: LLM Wiki ≠ RAG (do not confuse them)

This is the single most important conceptual anchor and I (a previous
session) slid off it once already.

> **RAG** does the synthesis work **at query time**. It chunks raw docs,
> retrieves nearest chunks per question, asks the LLM to glue them into
> an answer. Nothing accumulates between queries.
>
> **LLM Wiki** does the synthesis work **at ingest time**. The LLM reads
> a source once, writes structured, interlinked markdown pages that
> already encode entities, concepts, cross-references, and contradictions.
> Queries then **read those pre-written pages** rather than retrieve
> chunks. The wiki is a persistent, compounding artifact.

Implications when extending this project:

- **Do not add vector search / BM25 / embeddings** as "the way to scale
  query". That re-imports the RAG paradigm and discards the work the
  wiki paid for at ingest. If a future task feels like it needs RAG, the
  correct LLM-Wiki-native response is one of:
  - **Hierarchical index** (category-level index, drilled into on demand)
  - **Synthesis pages** (`overview.md`, `themes.md` — second-order pages
    that compress N first-order pages into a few)
  - **Split the wiki** into multiple domain-specific wikis with their own
    schemas (this is what "unit reuse via config" is for)
- **The current `_query` reads only `index.md` (page titles).** That's
  weak, but the right fix is to make it read **page content** for the 2-5
  most relevant pages, not to bolt on retrieval. Implement as a small
  2-step graph: `select_pages(index) → read_pages → answer`. All
  full-text, no chunks.
- When in doubt, re-read the spirit: "the wiki IS the synthesis." Reads
  are cheap and dumb; ingest is where the thinking happens.

There ARE cases where RAG genuinely wins (exact quote retrieval,
needle-in-haystack over raw sources). Don't pretend LLM Wiki is a
superset. If such a use case appears, run a **separate** RAG over
`raw/` alongside the wiki, not instead of it.

---

## Where work currently stands

### Done
- Bus, WorkingUnit ABC, Message protocol with `trace_id` for LangSmith
- `SourceFetcher` and `SourceSplitter` (with packing) and `WikiMaintainer`
- LangSmith tracing wired (`run_config_from_message` in observability.py)
- LangGraph Studio entry via `langgraph.json` and module-level `graph`
- Multi-provider LLM config (default: DeepSeek; supports Anthropic/OpenAI)
- Dual-model setup: plan node uses `deepseek-v4-pro`, query uses
  `deepseek-v4-flash`
- Robust structured output: `method="json_mode"` (DeepSeek doesn't
  reliably support `json_schema`) + `IngestPlan.page_updates` Pydantic
  validator that falls back through `json.loads` → `dirtyjson.loads` +
  one automatic retry feeding the parse error back to the model + raw
  dump to `wiki/.debug/last_plan_failure.json` on terminal failure
- `_apply_sync` rejects paths missing `.md`, containing newlines, or
  escaping `wiki_dir`
- `_node_load_context`, `_node_apply`, `_node_index_and_log` all wrap
  blocking file IO in `asyncio.to_thread` (blockbuster in LangGraph dev
  enforces this)
- Tests: 6 passing (`tests/test_bus.py`, `tests/test_splitter.py`)
- Scripts: `scripts/ingest.py` (split → loop maintainer),
  `scripts/query.py`, `scripts/demo.py` (single-file ingest, kept as
  smoke test reference)

### Blocked / Open
- **Network allowlist**: previous session needed `api.deepseek.com` and
  `api.smith.langchain.com` added to the environment's network policy.
  User said they added them — **a new session is required for the
  policy to take effect**.
- **End-to-end ingest of the book has not yet succeeded.** The PDF
  `raw/Mediator Guide to Careers.PDF` is in place. Splitter outputs
  10 packed chapters (~13–15 KB each, estimate ~$0.22 total with
  DeepSeek). Last attempt failed with `Host not in allowlist`.
- **Linter unit** is the next planned unit (see below).

### Not started yet
- `Linter` unit (third planned unit)
- `Researcher` unit (fourth, optional)
- Actual CLI binary (`workunits ingest/query/lint`) — `scripts/` covers
  the same surface for now
- Query upgrade to read page content (see CRITICAL section above)

---

## Immediate next step for this new session

1. **Verify environment** in this order:
   ```bash
   git status                       # clean working tree expected
   echo $DEEPSEEK_API_KEY           # may be empty in a fresh container
   ls .env                          # gitignored; recreate if missing
   ```
   `.env` is **not in git**; it lives in the sandbox per session. Ask the
   user for the keys if env vars are not set (Anthropic, DeepSeek,
   LangSmith — all four were provided in the previous session).

2. **Re-run the book ingest** with the allowlist now in place:
   ```bash
   python scripts/ingest.py "Mediator Guide to Careers.PDF"
   ```
   Expected: 10 chapter ingests, ~5–10 minutes wall, ~$0.22 cost.
   Each chapter goes through `WikiMaintainer.ingest` and adds/updates
   pages in `wiki/`.

3. **Inspect the resulting wiki** before pushing — check `wiki/index.md`,
   `wiki/overview.md`, and a couple of entity/concept pages. The LLM
   should respect the schema in `schemas/book_wiki.md` (uses `parts/`
   instead of `chapters/`, INFP is central entity, etc.).

4. **Commit and push**.

5. **Then build the Linter unit** — the third unit in the original plan.
   It's deliberately LLM-Wiki-native (and has no equivalent in RAG):
   scan all wiki pages, find contradictions, orphan pages, important
   concepts mentioned but lacking their own page, stale claims. Output a
   health report and optional fix suggestions. Communicates back to
   WikiMaintainer for any auto-fixes the user approves.

---

## Things that broke in the previous session (so you don't repeat them)

1. **Push 403 from git proxy** — fix was installing the Claude GitHub
   App at https://github.com/apps/claude and granting access to the
   repo, then **starting a new session**. Same pattern for any future
   credential/permission issue.
2. **Blocking IO inside LangGraph nodes** — `blockbuster` in
   `langgraph dev` raises `BlockingError` on sync `mkdir`/`read_text`/
   `write_text`. Always wrap file IO in `asyncio.to_thread` inside async
   nodes. Already done; copy that pattern for new nodes.
3. **Claude tool-use stringification** — `with_structured_output`
   sometimes serializes nested lists as JSON-encoded strings, and
   sometimes the inner JSON is itself malformed (unescaped quotes in
   long markdown content). Fix path: prompt instruction + Pydantic
   `field_validator(mode="before")` with `json.loads → dirtyjson.loads`
   + one self-correcting retry + final raw-dump for debugging. Already
   in place; if you ever add another structured-output schema, apply
   the same belt-and-suspenders.
4. **Sandbox network allowlist** — repeatedly hit "Host not in
   allowlist" for `api.smith.langchain.com` and `api.deepseek.com`.
   Allowlist changes require a new session to take effect.
5. **pymupdf4llm makes every bold heading an H2** — for this book that
   yielded 136 H2s and 89 raw "chapters" of ~1.5 KB each. The splitter
   now greedy-packs consecutive small sections up to
   `target_chapter_chars` (default 15000). Resulted in 10 reasonable
   super-chapters. Test (`tests/test_splitter.py`) uses
   `target_chapter_chars=0` to disable packing for the assertion-style
   tests.

---

## Project layout cheat sheet

```
src/aiworkingunits/
├── __init__.py            # public exports
├── messages.py            # Message, MessageType, reply()
├── bus.py                 # AsyncMessageBus (in-process, asyncio.Queue per unit)
├── unit.py                # WorkingUnit ABC, UnitConfig (Pydantic)
├── observability.py       # load_env, run_config_from_message (trace correlation)
└── units/
    ├── source_fetcher.py
    ├── source_splitter.py
    └── wiki_maintainer.py # includes module-level `graph` for LangGraph Studio

schemas/book_wiki.md       # WikiMaintainer's domain config (page types, ingest rules)
scripts/
├── ingest.py              # canonical: split → ingest each chapter
├── query.py               # canonical: ask the wiki
└── demo.py                # legacy single-file ingest (kept as a fetcher smoke test)
tests/
├── test_bus.py            # 3 tests
└── test_splitter.py       # 3 tests
langgraph.json             # LangGraph Studio: points at wiki_maintainer.graph
pyproject.toml             # deps: langgraph, langchain-anthropic, langchain-openai,
                           #       langsmith, pydantic, pymupdf4llm, dirtyjson
.env.example               # template; .env is gitignored
raw/                       # source files (PDF / md); user-provided
wiki/                      # LLM-generated wiki output
```

### Provider config (in `WikiMaintainerConfig`)

| Field | Default | Notes |
|---|---|---|
| `llm_provider` | `"deepseek"` | `"anthropic"` / `"openai"` / `"deepseek"` |
| `llm_base_url` | None → provider default | DeepSeek uses `https://api.deepseek.com` |
| `model` | `"deepseek-v4-flash"` | Used by `wiki.query` |
| `plan_model` | `"deepseek-v4-pro"` | Used by `wiki.ingest`'s plan node |
| `structured_output_method` | `"json_mode"` | DeepSeek doesn't reliably support `json_schema` |
| `max_tokens` | 16384 | Output cap, plenty for ~15-page plan |
| `max_pages_per_ingest` | 15 | Schema-level cap |

### Common commands

```bash
# Install (editable)
pip install -e ".[dev]"

# Tests
python -m pytest -v

# Ingest a source
python scripts/ingest.py "Mediator Guide to Careers.PDF"

# Query the wiki
python scripts/query.py "your question here"

# LangGraph Studio (debug graph state, prompts, structured outputs)
langgraph dev   # then open the URL it prints

# LangSmith trace UI
# https://smith.langchain.com → project "aiworkingunits"
```

---

## Branch & workflow

- Branch: `claude/ai-agent-working-units-wfpk7`
- Commit style: descriptive subjects ("Wrap blocking IO …", "Switch
  plan node to json_schema …"), bodies that explain *why* not just
  *what*, ending with the session URL.
- Always commit + push between meaningful steps — sandbox state is
  ephemeral and the stop hook nags about unpushed commits.
- `.env` is gitignored and must be recreated per session with:
  ```
  ANTHROPIC_API_KEY=...
  DEEPSEEK_API_KEY=...
  LANGSMITH_API_KEY=...
  LANGSMITH_TRACING=true
  LANGSMITH_PROJECT=aiworkingunits
  ```

---

## Conventions worth keeping

- **No comments that restate code.** Comments only when the *why* is
  non-obvious (existing comments mostly explain known LLM failure modes;
  match that bar).
- **Defensive at boundaries, trusting inside.** Pydantic validators
  guard model output; internal Python code trusts its own data.
- **Every new unit gets:** an `*.py` file in `units/`, a Pydantic config
  subclass, a `make_*` factory with sensible defaults, a smoke test that
  uses the bus (no LLM mock needed — bus tests use plain `EchoUnit`s),
  and a capability namespace (`<thing>.<verb>`).
- **Async file IO** wrapped in `asyncio.to_thread` inside graph nodes
  (LangGraph dev's blockbuster enforces this).
- **Structured output** through Pydantic with the validator+retry+dump
  pattern. Never trust a single LLM JSON emission.

---

## A note on tone in this file

Future-you should be able to skim this in 60 seconds and know what to
do without re-reading the entire chat history. If something here is
out of date after your work, **update this file as part of that commit**
so the *next* session inherits a current map. This is itself a tiny
LLM Wiki — a compounding artifact that pays back the time spent
maintaining it.
