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
from pydantic import BaseModel, Field, field_validator

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

    # LLM provider config. The base UnitConfig.model field is used as the query
    # model; plan_model (if set) is used for the more demanding ingest plan.
    llm_provider: LLMProvider = "deepseek"
    llm_base_url: str | None = None
    structured_output_method: StructuredMethod = "json_mode"
    plan_model: str | None = None


class PageUpdate(BaseModel):
    path: str = Field(description="Path of the wiki page relative to wiki_dir, e.g. 'entities/alice.md'")
    action: Literal["create", "update", "append"] = Field(description="create new page, full update, or append section")
    content: str = Field(description="Full markdown content (for create/update) or section to append")
    rationale: str = Field(description="One sentence explaining why this change")


class IngestPlan(BaseModel):
    source_summary: str = Field(description="A 2-4 sentence summary of the source")
    page_updates: list[PageUpdate] = Field(description="Wiki page changes to apply for this source")

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
    plan: IngestPlan
    applied: list[str]


def _extract_plan(result: Any) -> IngestPlan | None:
    if isinstance(result, IngestPlan):
        return result
    if isinstance(result, dict):
        parsed = result.get("parsed")
        if isinstance(parsed, IngestPlan):
            return parsed
    return None


class WikiMaintainer(WorkingUnit):
    """Owns a wiki directory. Ingests sources via a LangGraph pipeline.

    Capabilities:
      - "wiki.ingest": payload {"source_path", "title", "content"}
      - "wiki.query": payload {"question"}  (simple v1: read index + cite pages)
    """

    config_cls = WikiMaintainerConfig
    config: WikiMaintainerConfig

    def __init__(self, config: WikiMaintainerConfig, bus: Any) -> None:
        super().__init__(config, bus)
        self.config.wiki_dir.mkdir(parents=True, exist_ok=True)
        self._llm_query = self._build_llm(config.model)
        self._llm_plan = self._build_llm(config.plan_model) if config.plan_model else self._llm_query
        self._graph = self._build_graph()

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
            answer = await self._query(msg.payload.get("question", ""), config)
            return msg.reply(payload={"answer": answer}, sender=self.unit_id)
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
            "summary": final_state.get("plan").source_summary if final_state.get("plan") else "",
        }

    async def _query(self, question: str, config: dict[str, Any]) -> str:
        index = await asyncio.to_thread(self._read_index)
        prompt = (
            "You are a wiki librarian. Use the index below to answer the question."
            " Cite pages by their relative path.\n\n"
            f"# Wiki Index\n{index}\n\n# Question\n{question}"
        )
        result = await self._llm_query.ainvoke([HumanMessage(content=prompt)], config=config)
        return str(result.content)

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
        def _read() -> tuple[str, str]:
            return self._read_text(self.config.schema_path), self._read_index()

        schema, index = await asyncio.to_thread(_read)
        return {"schema_doc": schema, "index_doc": index}

    async def _node_plan(self, state: WikiState) -> dict[str, Any]:
        system = SystemMessage(
            content=(
                "You are the maintainer of a personal wiki built from sources."
                " You are given the wiki schema, the current index, and a new source."
                " Produce a plan of page updates that integrates the source into the wiki."
                " Update existing pages when possible; create new pages when needed."
                f" Limit yourself to at most {self.config.max_pages_per_ingest} page updates."
                " Always include a source summary page under sources/.\n\n"
                "CRITICAL FORMAT RULES:\n"
                "- `page_updates` MUST be a JSON array of objects, NEVER a string containing JSON.\n"
                "- Inside each `content` field, write plain markdown with real newlines."
                " Do not pre-escape newlines, quotes, or backslashes — the tool layer escapes them for you.\n"
                "- Keep each `content` field self-contained markdown for that single page."
            )
        )
        user = HumanMessage(
            content=(
                f"# Wiki schema\n{state.get('schema_doc','')}\n\n"
                f"# Current index\n{state.get('index_doc','(empty)')}\n\n"
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

    async def _node_index_and_log(self, state: WikiState) -> dict[str, Any]:
        await asyncio.to_thread(self._index_and_log_sync, state)
        return {}

    def _index_and_log_sync(self, state: WikiState) -> None:
        applied = state.get("applied", [])
        title = state.get("source_title", "")
        plan = state.get("plan")
        summary = plan.source_summary if plan else ""

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
        capabilities=["wiki.ingest", "wiki.query"],
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
