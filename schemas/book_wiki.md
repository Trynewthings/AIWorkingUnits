# Wiki Schema: Book Companion

This document configures the WikiMaintainer for ingesting a single book
(chapter by chapter, or as a whole PDF). Edit it to retarget the same
unit at a different domain.

## Directory layout

```
wiki/
├── index.md            # auto-rebuilt catalog of all pages
├── log.md              # append-only chronicle of ingests/queries/lints
├── overview.md         # evolving thesis/synthesis of the whole book
├── sources/            # one summary page per ingested source (chapter, etc.)
│   └── ch01-<slug>.md
├── entities/           # people, places, organizations, artifacts
│   └── <slug>.md
├── concepts/           # ideas, themes, arguments, motifs
│   └── <slug>.md
└── chapters/           # one page per chapter with structural notes
    └── ch01.md
```

## Page conventions

- Filenames are lowercase, hyphen-separated slugs (`alice-liddell.md`).
- Every page starts with a `# Title` H1.
- Cross-links use markdown links with the relative path: `[Alice](../entities/alice.md)`.
- Optional YAML frontmatter is allowed (`tags`, `aliases`, `first_seen_in`).
- When a claim is sourced, append a `> source: sources/<slug>.md` blockquote near it.

## Page types and what each must contain

### sources/<slug>.md
A summary of one ingested source.
Sections: `## Summary`, `## Key claims`, `## Entities mentioned`, `## Concepts touched`, `## Quotes`.

### entities/<slug>.md
Sections: `## Description`, `## Appearances` (list of source links), `## Relationships`.

### concepts/<slug>.md
Sections: `## Definition`, `## Evidence` (links to sources/quotes), `## Open questions`.

### chapters/<slug>.md
Sections: `## Setting`, `## Events`, `## Notable passages`, `## Entities introduced`.

### overview.md
A living synthesis: `## Thesis`, `## Plot/Argument arc`, `## Open threads`, `## Contradictions noted`.

## Ingest workflow

When a new source arrives:

1. Write `sources/<slug>.md` with summary, key claims, mentioned entities/concepts.
2. For each entity mentioned: create or update `entities/<slug>.md` (add to `## Appearances`).
3. For each concept touched: create or update `concepts/<slug>.md` (add evidence).
4. If the source is a chapter: create or update the corresponding `chapters/<slug>.md`.
5. Update `overview.md` only when this source meaningfully shifts the thesis or arc.
6. The LangGraph node will rebuild `index.md` and append to `log.md` automatically — do not write those directly.

## Update rules

- Prefer updating existing pages over creating duplicates. Search the index first.
- When a new source contradicts an older claim, do NOT overwrite — add a `## Contradictions` section noting both sides with citations.
- Cap each ingest at 15 page updates; pick the most important ones.

## Domain hint (edit me)

> The current book is: **(set this when the PDF is uploaded)**.
> Genre: **(fiction | non-fiction | technical)**.
> Emphasize: **(plot/characters | arguments/evidence | concepts/definitions)**.
