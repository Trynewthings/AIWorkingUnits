from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator, model_validator

from aiworkingunits.messages import Message, MessageType
from aiworkingunits.observability import run_config_from_message
from aiworkingunits.unit import UnitConfig, WorkingUnit

logger = logging.getLogger(__name__)


LLMProvider = Literal["anthropic", "openai", "deepseek"]
StructuredMethod = Literal["function_calling", "json_mode", "json_schema"]

_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {"key_env": "ANTHROPIC_API_KEY"},
    "openai": {"key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
    "deepseek": {"key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com"},
}


class WikiMaintainerConfig(UnitConfig):
    wiki_dir: Path = Path("wiki")
    schema_path: Path = Path("schemas/book_wiki.md")
    max_pages_per_ingest: int = 15
    # Total character budget for the existing-wiki snapshot fed into plan.
    # Defends against runaway prompts as the wiki grows; pages are included
    # whole-or-not (no mid-page truncation) until the budget is exhausted.
    wiki_snapshot_char_budget: int = 60000
    # Maximum number of wiki pages the query graph will read into the answer
    # context. Selecting more rarely improves quality and burns tokens.
    query_max_pages: int = 5

    # LLM provider config. The base UnitConfig.model field is used as the query
    # model; plan_model (if set) is used for the more demanding ingest plan.
    llm_provider: LLMProvider = "deepseek"
    llm_base_url: str | None = None
    structured_output_method: StructuredMethod = "function_calling"
    plan_model: str | None = None


class PageUpdate(BaseModel):
    path: str = Field(description="Path of the wiki page relative to wiki_dir, e.g. 'entities/alice.md'")
    content: str = Field(description="Full markdown content (for create/update) or section to append")
    action: Literal["create", "update", "append"] = Field(
        default="create",
        description="create new page, full update, or append section",
    )
    rationale: str = Field(default="", description="One sentence explaining why this change")


class IngestPlan(BaseModel):
    synopsis: str = Field(
        description=(
            "A short (2-4 sentence) plain-text synopsis of THIS ingest for the log."
            " This is NOT a wiki page. Do not put markdown, headings, or page objects here."
            " Wiki page content always goes inside page_updates."
        )
    )
    page_updates: list[PageUpdate] = Field(
        description=(
            "Wiki page changes to apply for this source, including the source summary page"
            " (which lives at sources/<slug>.md inside page_updates, NOT in the synopsis field)."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def _recover_misplaced_fields(cls, data: Any) -> Any:
        """Defenses against common DeepSeek/Claude shape errors:

        - source_summary as a legacy field name → rename to synopsis
        - synopsis as a PageUpdate-shaped dict → move it into page_updates and
          derive a plain string synopsis from its content
        """
        if not isinstance(data, dict):
            return data
        if "synopsis" not in data and "source_summary" in data:
            data["synopsis"] = data.pop("source_summary")
        syn = data.get("synopsis")
        if isinstance(syn, dict) and "content" in syn and "path" in syn:
            page_updates = list(data.get("page_updates") or [])
            misplaced = dict(syn)
            misplaced.setdefault("action", "create")
            misplaced.setdefault("rationale", "auto-recovered from synopsis field")
            page_updates.insert(0, misplaced)
            data["page_updates"] = page_updates
            content = str(syn.get("content", ""))
            derived = next(
                (
                    ln.strip()
                    for ln in content.splitlines()
                    if ln.strip() and not ln.lstrip().startswith(("#", "-", "*", ">", "|"))
                ),
                "Source ingested (synopsis auto-derived from page content).",
            )
            data["synopsis"] = derived[:400]
        return data

    # Claude with structured output sometimes emits the list as a JSON-encoded
    # string and sometimes that string has unescaped quotes / backslashes inside
    # long markdown content. Try strict json first, then dirtyjson which tolerates
    # the common LLM JSON sins (unescaped quotes, trailing commas, raw newlines).
    @field_validator("page_updates", mode="before")
    @classmethod
    def _coerce_json_string(cls, v: Any) -> Any:
        if not isinstance(v, str):
            return v
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            import dirtyjson

            return list(dirtyjson.loads(v))


class WikiState(TypedDict, total=False):
    source_path: str
    source_title: str
    source_content: str
    schema_doc: str
    index_doc: str
    wiki_snapshot: str
    plan: IngestPlan
    applied: list[str]


class QuerySelection(BaseModel):
    """Plan output of the query graph's selection step."""

    selected_pages: list[str] = Field(
        description=(
            "Relative paths (from wiki_dir) of the 2-5 wiki pages most likely to"
            " contain the answer. Only return paths that appear verbatim in the index."
        )
    )
    reasoning: str = Field(
        default="",
        description="One sentence explaining the selection — used for transparency, not displayed to end users.",
    )


class WikiQueryState(TypedDict, total=False):
    question: str
    index_doc: str
    selection: QuerySelection
    page_bodies: dict[str, str]
    cited_pages: list[str]
    answer: str


def _extract_plan(result: Any) -> IngestPlan | None:
    if isinstance(result, IngestPlan):
        return result
    if isinstance(result, dict):
        parsed = result.get("parsed")
        if isinstance(parsed, IngestPlan):
            return parsed
    return None


def _extract_selection(result: Any) -> QuerySelection | None:
    if isinstance(result, QuerySelection):
        return result
    if isinstance(result, dict):
        parsed = result.get("parsed")
        if isinstance(parsed, QuerySelection):
            return parsed
    return None


class WikiMaintainer(WorkingUnit):
    """Owns a wiki directory. Ingests sources via a LangGraph pipeline.

    Capabilities:
      - "wiki.ingest": payload {"source_path", "title", "content"}
      - "wiki.query":  payload {"question"} →
            {"answer": str, "cited_pages": list[str], "reasoning": str}
        Two-phase: select 2-5 most-relevant pages from the index, read their
        full content, then answer using only that content (no chunking, no
        retrieval — the synthesis was already done at ingest time).
      - "wiki.repair": payload {"issues", "mode", "policies"} — see _repair
    """

    config_cls = WikiMaintainerConfig
    config: WikiMaintainerConfig

    def __init__(self, config: WikiMaintainerConfig, bus: Any) -> None:
        super().__init__(config, bus)
        self.config.wiki_dir.mkdir(parents=True, exist_ok=True)
        self._llm_query = self._build_llm(config.model)
        self._llm_plan = self._build_llm(config.plan_model) if config.plan_model else self._llm_query
        self._graph = self._build_graph()
        self._query_graph = self._build_query_graph()

    def _build_llm(self, model: str) -> BaseChatModel:
        provider = self.config.llm_provider
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        if provider in {"openai", "deepseek"}:
            from langchain_openai import ChatOpenAI

            defaults = _PROVIDER_DEFAULTS[provider]
            base_url = self.config.llm_base_url or defaults["base_url"]
            # Key may be absent at module-import time (e.g., when LangGraph Studio
            # loads the graph factory before .env is loaded). Defer the auth
            # failure to the actual API call rather than blocking construction.
            api_key = os.environ.get(defaults["key_env"], "__missing__")
            return ChatOpenAI(
                model=model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                base_url=base_url,
                api_key=api_key,
            )
        raise ValueError(f"unsupported llm_provider: {provider}")

    async def handle(self, msg: Message) -> Message | None:
        if msg.type != MessageType.REQUEST:
            return None
        cap = msg.capability or msg.payload.get("op")
        if cap == "wiki.ingest":
            config = run_config_from_message(msg, unit_id=self.unit_id, run_name=f"{self.unit_id}.ingest")
            result = await self._ingest(msg.payload, config)
            return msg.reply(payload=result, sender=self.unit_id)
        if cap == "wiki.query":
            config = run_config_from_message(msg, unit_id=self.unit_id, run_name=f"{self.unit_id}.query")
            result = await self._query(msg.payload.get("question", ""), config)
            return msg.reply(payload=result, sender=self.unit_id)
        if cap == "wiki.repair":
            result = await asyncio.to_thread(self._repair, msg.payload)
            return msg.reply(payload=result, sender=self.unit_id)
        return None

    async def _ingest(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        state: WikiState = {
            "source_path": payload.get("source_path", ""),
            "source_title": payload.get("title", ""),
            "source_content": payload.get("content", ""),
        }
        final_state = await self._graph.ainvoke(state, config=config)
        return {
            "applied_pages": final_state.get("applied", []),
            "summary": final_state.get("plan").synopsis if final_state.get("plan") else "",
        }

    async def _query(self, question: str, config: dict[str, Any]) -> dict[str, Any]:
        if not question.strip():
            return {"answer": "(empty question)", "cited_pages": [], "reasoning": ""}
        state: WikiQueryState = {"question": question.strip()}
        final = await self._query_graph.ainvoke(state, config=config)
        return {
            "answer": final.get("answer", ""),
            "cited_pages": final.get("cited_pages", []),
            "reasoning": final.get("selection").reasoning if final.get("selection") else "",
        }

    def _build_query_graph(self) -> Any:
        graph: StateGraph = StateGraph(WikiQueryState)
        graph.add_node("load_index", self._qnode_load_index)
        graph.add_node("select_pages", self._qnode_select_pages)
        graph.add_node("read_pages", self._qnode_read_pages)
        graph.add_node("answer", self._qnode_answer)
        graph.add_edge(START, "load_index")
        graph.add_edge("load_index", "select_pages")
        graph.add_edge("select_pages", "read_pages")
        graph.add_edge("read_pages", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    async def _qnode_load_index(self, state: WikiQueryState) -> dict[str, Any]:
        index = await asyncio.to_thread(self._read_index)
        return {"index_doc": index}

    async def _qnode_select_pages(self, state: WikiQueryState) -> dict[str, Any]:
        system = SystemMessage(
            content=(
                "You are a wiki librarian. You read the index of an LLM-maintained"
                " wiki and pick the small set of pages most likely to contain the"
                " answer to a question. Respond as JSON matching the QuerySelection"
                " schema with fields `selected_pages` (list of relative paths) and"
                " `reasoning` (one sentence).\n\n"
                f"Rules:\n"
                f"- Return between 1 and {self.config.query_max_pages} page paths.\n"
                "- Each path MUST appear verbatim in the index (copy-paste, do not invent).\n"
                "- Prefer entity / concept / part pages over source-summary pages when"
                " both could answer; source pages are mainly evidence trails.\n"
                "- If the question is too broad, pick a small set of overview pages"
                " (e.g. overview.md, index-like category pages).\n"
                "- If the question is unanswerable from the index alone (e.g. asks"
                " about content the wiki clearly does not cover), still return your"
                " best 1-2 guesses so the answer node can confirm absence."
            )
        )
        user = HumanMessage(
            content=(
                f"# Question\n{state['question']}\n\n"
                f"# Wiki index\n{state.get('index_doc', '(empty)')}"
            )
        )
        structured = self._llm_query.with_structured_output(
            QuerySelection, method=self.config.structured_output_method, include_raw=True
        )
        result = await structured.ainvoke([system, user])
        selection = _extract_selection(result)
        if selection is None:
            logger.warning("query select_pages: structured output failed, falling back to overview-only")
            selection = QuerySelection(
                selected_pages=["overview.md"], reasoning="fallback: selection parse failed"
            )
        return {"selection": selection}

    async def _qnode_read_pages(self, state: WikiQueryState) -> dict[str, Any]:
        selection = state.get("selection")
        if selection is None:
            return {"page_bodies": {}, "cited_pages": []}
        chosen = list(selection.selected_pages)[: self.config.query_max_pages]
        bodies = await asyncio.to_thread(self._read_selected_pages, chosen)
        return {"page_bodies": bodies, "cited_pages": list(bodies.keys())}

    def _read_selected_pages(self, paths: list[str]) -> dict[str, str]:
        """Validate paths and load contents. Drops invalid/missing paths with a log line."""
        wiki_root = self.config.wiki_dir.resolve()
        out: dict[str, str] = {}
        for raw in paths:
            rel = raw.strip().lstrip("/")
            if not rel.endswith(".md") or "\n" in rel or "\x00" in rel:
                logger.warning("query: rejected non-markdown or malformed path %r", raw)
                continue
            target = (self.config.wiki_dir / rel).resolve()
            try:
                target.relative_to(wiki_root)
            except ValueError:
                logger.warning("query: rejected page outside wiki_dir: %s", rel)
                continue
            if not target.exists():
                logger.warning("query: selector picked non-existent page: %s", rel)
                continue
            out[rel] = target.read_text(encoding="utf-8")
        return out

    async def _qnode_answer(self, state: WikiQueryState) -> dict[str, Any]:
        bodies = state.get("page_bodies", {})
        if not bodies:
            return {
                "answer": (
                    "I could not find any wiki page that addresses this question."
                    " (The selector returned no valid pages.) Consider rephrasing or"
                    " ingesting a source that covers the topic."
                )
            }
        joined = "\n\n".join(
            f"## Page: {rel}\n{body.strip()}" for rel, body in bodies.items()
        )
        system = SystemMessage(
            content=(
                "You are answering a user's question using only the wiki pages provided."
                " Rules:\n"
                "- Answer in the same language as the question.\n"
                "- Cite the pages you use inline as `(see <relative-path>)`.\n"
                "- If the pages do not contain the answer, say so explicitly rather than"
                " guessing. Suggest which kind of source the user would need to ingest.\n"
                "- Be concise. Don't restate the question."
            )
        )
        user = HumanMessage(
            content=(
                f"# Question\n{state['question']}\n\n"
                f"# Relevant wiki pages\n{joined}"
            )
        )
        result = await self._llm_query.ainvoke([system, user])
        return {"answer": str(result.content)}

    def _build_graph(self) -> Any:
        graph: StateGraph = StateGraph(WikiState)
        graph.add_node("load_context", self._node_load_context)
        graph.add_node("plan", self._node_plan)
        graph.add_node("apply", self._node_apply)
        graph.add_node("update_index_and_log", self._node_index_and_log)
        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "plan")
        graph.add_edge("plan", "apply")
        graph.add_edge("apply", "update_index_and_log")
        graph.add_edge("update_index_and_log", END)
        return graph.compile()

    async def _node_load_context(self, state: WikiState) -> dict[str, Any]:
        def _read() -> tuple[str, str, str]:
            return (
                self._read_text(self.config.schema_path),
                self._read_index(),
                self._read_wiki_snapshot(),
            )

        schema, index, snapshot = await asyncio.to_thread(_read)
        return {"schema_doc": schema, "index_doc": index, "wiki_snapshot": snapshot}

    def _read_wiki_snapshot(self) -> str:
        """Concatenate existing wiki pages (except index/log) into one string.

        Pages are included whole, never mid-page-truncated, until the configured
        char budget is exhausted. This gives the plan node the actual current
        content of pages it may update, preventing the LLM from emitting
        '(existing content here)' placeholders for sections it doesn't intend
        to change.
        """
        budget = self.config.wiki_snapshot_char_budget
        if budget <= 0:
            return "(snapshot disabled)"
        chunks: list[str] = []
        used = 0
        omitted = 0
        for p in sorted(self.config.wiki_dir.rglob("*.md")):
            rel = p.relative_to(self.config.wiki_dir).as_posix()
            if rel in {"index.md", "log.md"} or rel.startswith(".debug/"):
                continue
            body = p.read_text(encoding="utf-8")
            block = f"\n\n----- BEGIN PAGE: {rel} -----\n{body.rstrip()}\n----- END PAGE: {rel} -----"
            if used + len(block) > budget:
                omitted += 1
                continue
            chunks.append(block)
            used += len(block)
        if not chunks:
            return "(no existing pages)"
        header = f"({len(chunks)} pages included, {omitted} omitted due to budget)\n"
        return header + "".join(chunks)

    async def _node_plan(self, state: WikiState) -> dict[str, Any]:
        system = SystemMessage(
            content=(
                "You are the maintainer of a personal wiki built from sources."
                " You are given the wiki schema, the current index, the FULL CURRENT"
                " CONTENT of existing pages, and a new source."
                " Produce a plan that integrates the source into the wiki."
                " Update existing pages when possible; create new pages when needed."
                f" Limit yourself to at most {self.config.max_pages_per_ingest} page updates.\n\n"
                "OUTPUT SHAPE (be exact, this is the most common failure point):\n"
                "- `synopsis`: a short PLAIN-TEXT string (2-4 sentences) summarizing"
                " what this ingest does. NOT a markdown page. NOT a page object."
                " It goes into the log so humans can skim what happened.\n"
                "- `page_updates`: a JSON ARRAY of page-update objects (never a string)."
                " Every entry MUST have `path`, `content`, `action`, `rationale`.\n"
                "- One of the page_updates MUST be the source summary page at"
                " `sources/<slug>.md` — that page lives inside page_updates,"
                " NOT in the synopsis field.\n\n"
                "CRITICAL RULE FOR action=update (this is how pages get silently destroyed):\n"
                "- `update` overwrites the entire page. You MUST output the COMPLETE new"
                " page body, including verbatim any sections you do not intend to change.\n"
                "- The existing page content is provided to you below — copy unchanged"
                " sections into your output exactly as written.\n"
                "- NEVER use placeholders like '(existing content here)', '(unchanged)',"
                " '(see previous)', or '...' — they will be written to disk literally"
                " and destroy real content.\n"
                "- If you only want to add to a page without rewriting it, use"
                " `action=append` instead — that concatenates a new section to the end\n\n"
                "Inside each `content` field, write plain markdown with real newlines."
                " Do not pre-escape newlines, quotes, or backslashes — the tool layer"
                " escapes them for you. Keep each `content` self-contained for one page."
            )
        )
        user = HumanMessage(
            content=(
                f"# Wiki schema\n{state.get('schema_doc','')}\n\n"
                f"# Current index\n{state.get('index_doc','(empty)')}\n\n"
                f"# Existing pages (full content)\n{state.get('wiki_snapshot','(none)')}\n\n"
                f"# New source: {state.get('source_title','')}\n"
                f"Path (in raw): {state.get('source_path','')}\n\n"
                f"## Source content\n{state.get('source_content','')[:60000]}"
            )
        )
        # Configured method per provider. json_schema is strictest (where supported);
        # json_mode is the OpenAI-compatible fallback that DeepSeek supports. Either
        # way, our Pydantic validator + dirtyjson + retry layer catches the rest.
        structured = self._llm_plan.with_structured_output(
            IngestPlan, method=self.config.structured_output_method, include_raw=True
        )
        result = await structured.ainvoke([system, user])
        plan = _extract_plan(result)
        if plan is not None:
            return {"plan": plan}

        # One retry: feed the parse error back so the model can self-correct.
        parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
        logger.warning("plan parse failed on attempt 1, retrying with error feedback: %r", parsing_error)
        retry_system = SystemMessage(
            content=(
                system.content
                + "\n\nYour previous response could not be parsed. Parse error:\n"
                + repr(parsing_error)
                + "\n\nProduce the SAME plan again, but this time double-check that"
                " `page_updates` is a real JSON array of objects (not a string),"
                " and that every string field is properly escaped."
            )
        )
        retry = await structured.ainvoke([retry_system, user])
        plan = _extract_plan(retry)
        if plan is not None:
            logger.info("plan parsed on retry")
            return {"plan": plan}

        await asyncio.to_thread(self._dump_failed_plan, retry)
        err = retry.get("parsing_error") if isinstance(retry, dict) else None
        raise RuntimeError(
            "plan node failed both attempts; raw response dumped to wiki/.debug/last_plan_failure.json"
        ) from err

    def _dump_failed_plan(self, result: Any) -> None:
        debug_dir = self.config.wiki_dir / ".debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / "last_plan_failure.json"
        raw = result.get("raw") if isinstance(result, dict) else None
        snapshot = {
            "parsing_error": repr(result.get("parsing_error")) if isinstance(result, dict) else None,
            "raw_content": getattr(raw, "content", None),
            "raw_tool_calls": getattr(raw, "tool_calls", None),
        }
        path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
        logger.error("plan failure dumped to %s", path)

    async def _node_apply(self, state: WikiState) -> dict[str, Any]:
        plan: IngestPlan = state["plan"]
        applied = await asyncio.to_thread(self._apply_sync, plan)
        return {"applied": applied}

    def _apply_sync(self, plan: IngestPlan) -> list[str]:
        applied: list[str] = []
        for upd in plan.page_updates:
            rel = upd.path.strip().lstrip("/")
            if not rel.endswith(".md") or "\n" in rel:
                logger.warning("rejected non-markdown or malformed path: %r", upd.path)
                continue
            target = (self.config.wiki_dir / rel).resolve()
            try:
                target.relative_to(self.config.wiki_dir.resolve())
            except ValueError:
                logger.warning("rejected page outside wiki_dir: %s", rel)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if upd.action == "append" and target.exists():
                existing = target.read_text(encoding="utf-8")
                target.write_text(existing.rstrip() + "\n\n" + upd.content.strip() + "\n", encoding="utf-8")
            else:
                target.write_text(upd.content.strip() + "\n", encoding="utf-8")
            applied.append(rel)
        return applied

    # Default repair policies. `broken_link` defaults to skip because removing
    # a dead link is a real choice (user may instead want to create the missing
    # page). Caller can opt in by passing policies={"broken_link": "fix"}.
    _REPAIR_DEFAULTS: dict[str, str] = {
        "missing_h1": "fix",
        "broken_link": "skip",
        "orphan_page": "skip",
        "placeholder_text": "skip",
        "stub_page": "skip",
    }

    # Inline copy of the markdown-link regex (shared shape with WikiLinter).
    # Kept local to avoid cross-unit imports between sibling units.
    _LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s#]+(?:#[^)\s]+)?)\)")

    def _repair(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply (or suggest) deterministic fixes for a list of lint issues.

        payload:
          - issues: list of lint-issue dicts (see WikiLinter.LintIssue)
          - mode: "suggest" (default) → return proposals without writing
                  "apply" → write changes to disk
          - policies: dict[issue_type, "fix" | "skip"] — overrides defaults

        Returns: {proposals: [...], summary: {considered, fixed, skipped, errors}}
        """
        issues: list[dict[str, Any]] = list(payload.get("issues", []))
        mode = payload.get("mode", "suggest")
        if mode not in {"suggest", "apply"}:
            raise ValueError(f"unsupported repair mode: {mode!r}")
        policies = dict(self._REPAIR_DEFAULTS)
        policies.update(payload.get("policies") or {})

        # Group enabled issues by path so each file is read and written at most once.
        by_path: dict[str, list[dict[str, Any]]] = {}
        skipped: list[dict[str, Any]] = []
        for issue in issues:
            itype = issue.get("type", "")
            ipath = issue.get("path", "")
            if policies.get(itype, "skip") != "fix":
                skipped.append({**issue, "reason": f"policy={policies.get(itype, 'skip')}"})
                continue
            if itype not in {"missing_h1", "broken_link"}:
                # Recognized policy=fix but no fixer implemented yet.
                skipped.append({**issue, "reason": "no fixer implemented for this issue type"})
                continue
            by_path.setdefault(ipath, []).append(issue)

        proposals: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for rel, page_issues in by_path.items():
            try:
                proposals.extend(self._repair_one_page(rel, page_issues, mode))
            except Exception as e:
                logger.exception("repair failed for %s", rel)
                errors.append({"path": rel, "error": str(e), "error_type": type(e).__name__})

        fixed = sum(1 for p in proposals if p.get("applied") or mode == "suggest" and p.get("would_change"))
        return {
            "proposals": proposals,
            "skipped": skipped,
            "summary": {
                "considered": len(issues),
                "fixed": fixed if mode == "apply" else 0,
                "proposed": sum(1 for p in proposals if p.get("would_change")) if mode == "suggest" else 0,
                "skipped": len(skipped),
                "errors": len(errors),
            },
            "errors": errors,
            "mode": mode,
        }

    def _repair_one_page(
        self, rel: str, page_issues: list[dict[str, Any]], mode: str
    ) -> list[dict[str, Any]]:
        rel_clean = rel.strip().lstrip("/")
        if not rel_clean.endswith(".md") or "\n" in rel_clean:
            return [{"path": rel, "applied": False, "would_change": False,
                     "reason": "malformed path", "issue_types": [i["type"] for i in page_issues]}]
        target = (self.config.wiki_dir / rel_clean).resolve()
        try:
            target.relative_to(self.config.wiki_dir.resolve())
        except ValueError:
            return [{"path": rel, "applied": False, "would_change": False,
                     "reason": "path escapes wiki_dir", "issue_types": [i["type"] for i in page_issues]}]
        if not target.exists():
            return [{"path": rel, "applied": False, "would_change": False,
                     "reason": "page no longer exists"}]

        original = target.read_text(encoding="utf-8")
        new_content = original
        actions: list[dict[str, Any]] = []

        # Run fixers in a deterministic order: structural first, then content.
        if any(i["type"] == "missing_h1" for i in page_issues):
            new_content, info = self._fix_missing_h1(rel_clean, new_content)
            if info:
                actions.append(info)
        if any(i["type"] == "broken_link" for i in page_issues):
            new_content, info = self._fix_broken_links(rel_clean, new_content)
            if info:
                actions.append(info)

        would_change = new_content != original
        applied = False
        if would_change and mode == "apply":
            target.write_text(new_content, encoding="utf-8")
            applied = True
        return [{
            "path": rel_clean,
            "actions": actions,
            "would_change": would_change,
            "applied": applied,
            "before_excerpt": original[:200],
            "after_excerpt": new_content[:200],
        }]

    def _fix_missing_h1(self, rel: str, body: str) -> tuple[str, dict[str, Any] | None]:
        if body.lstrip().startswith("# "):
            return body, None  # false positive — page already has H1
        title = self._title_from_path(rel)
        new = f"# {title}\n\n" + body.lstrip()
        return new, {"action": "prepend_h1", "title": title}

    def _fix_broken_links(self, rel: str, body: str) -> tuple[str, dict[str, Any] | None]:
        wiki_root = self.config.wiki_dir.resolve()
        page_dir = (self.config.wiki_dir / rel).resolve().parent
        removed: list[dict[str, str]] = []

        def _replacer(match: re.Match[str]) -> str:
            text, target = match.group(1), match.group(2).split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:", "ftp://")):
                return match.group(0)
            if not target.endswith(".md"):
                return match.group(0)
            try:
                resolved = (page_dir / target).resolve()
                resolved.relative_to(wiki_root)
            except (ValueError, OSError):
                removed.append({"text": text, "target": target, "reason": "escapes wiki_dir"})
                return text
            if resolved.exists():
                return match.group(0)
            removed.append({"text": text, "target": target, "reason": "target does not exist"})
            return text

        new_body = self._LINK_RE.sub(_replacer, body)
        if not removed:
            return body, None
        return new_body, {"action": "delinkify", "removed": removed}

    @staticmethod
    def _title_from_path(rel: str) -> str:
        stem = Path(rel).stem
        return " ".join(part.capitalize() for part in stem.replace("_", "-").split("-") if part)

    async def _node_index_and_log(self, state: WikiState) -> dict[str, Any]:
        await asyncio.to_thread(self._index_and_log_sync, state)
        return {}

    def _index_and_log_sync(self, state: WikiState) -> None:
        applied = state.get("applied", [])
        title = state.get("source_title", "")
        plan = state.get("plan")
        summary = plan.synopsis if plan else ""

        log_path = self.config.wiki_dir / "log.md"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entry = (
            f"## [{ts}] ingest | {title}\n"
            f"- source: {state.get('source_path','')}\n"
            f"- pages touched: {len(applied)}\n"
            f"- summary: {summary}\n"
        )
        existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Log\n\n"
        log_path.write_text(existing.rstrip() + "\n\n" + entry, encoding="utf-8")

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        index_path = self.config.wiki_dir / "index.md"
        entries: dict[str, list[tuple[str, str]]] = {}
        for p in sorted(self.config.wiki_dir.rglob("*.md")):
            rel = p.relative_to(self.config.wiki_dir).as_posix()
            if rel in {"index.md", "log.md"}:
                continue
            category = rel.split("/", 1)[0] if "/" in rel else "root"
            first_line = next((ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()), "")
            title = re.sub(r"^#+\s*", "", first_line) or p.stem
            entries.setdefault(category, []).append((rel, title))

        lines = ["# Index", ""]
        for cat in sorted(entries):
            lines.append(f"## {cat}")
            for rel, title in entries[cat]:
                lines.append(f"- [{title}]({rel})")
            lines.append("")
        index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _read_index(self) -> str:
        index_path = self.config.wiki_dir / "index.md"
        if not index_path.exists():
            return "(empty)"
        return index_path.read_text(encoding="utf-8")

    @staticmethod
    def _read_text(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")


def make_maintainer(
    bus: Any,
    *,
    unit_id: str = "wiki_maintainer",
    wiki_dir: Path | str = "wiki",
    schema_path: Path | str = "schemas/book_wiki.md",
    model: str = "deepseek-v4-flash",
    plan_model: str | None = "deepseek-v4-pro",
    llm_provider: LLMProvider = "deepseek",
    structured_output_method: StructuredMethod = "json_mode",
) -> WikiMaintainer:
    cfg = WikiMaintainerConfig(
        unit_id=unit_id,
        capabilities=["wiki.ingest", "wiki.query", "wiki.repair"],
        wiki_dir=Path(wiki_dir),
        schema_path=Path(schema_path),
        model=model,
        plan_model=plan_model,
        llm_provider=llm_provider,
        structured_output_method=structured_output_method,
    )
    return WikiMaintainer(cfg, bus)


# Module-level compiled graph exposed for LangGraph Studio (langgraph.json).
# Studio invokes this graph directly with a WikiState dict (no bus); it works
# but skips the cross-unit message correlation that the WorkingUnit wrapper
# provides. Use Studio for graph-internal debugging, LangSmith for cross-unit.
def _build_studio_graph() -> Any:
    from aiworkingunits.bus import AsyncMessageBus  # noqa: WPS433 — avoid import cycle at top

    repo_root = Path(__file__).resolve().parents[3]
    return make_maintainer(
        AsyncMessageBus(),
        wiki_dir=repo_root / "wiki",
        schema_path=repo_root / "schemas" / "book_wiki.md",
    )._graph


graph = _build_studio_graph()
