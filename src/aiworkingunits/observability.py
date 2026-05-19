from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aiworkingunits.messages import Message


def load_env(repo_root: Path | None = None) -> None:
    """Load .env so LangChain/LangSmith pick up API keys.

    Must be called before importing/instantiating any LangChain LLM if you
    want LANGSMITH_TRACING=true to take effect for that process.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if repo_root is None:
        repo_root = Path.cwd()
    load_dotenv(repo_root / ".env")


def run_config_from_message(msg: Message, *, unit_id: str, run_name: str | None = None) -> dict[str, Any]:
    """Build a LangChain RunnableConfig that ties this graph run to the bus trace.

    Threading trace_id through every cross-unit hop is what lets LangSmith
    group fetcher.handle + maintainer.ingest + linter.scan into one trace.
    """
    return {
        "metadata": {
            "trace_id": msg.trace_id,
            "correlation_id": msg.id,
            "sender_unit": msg.sender,
            "receiver_unit": unit_id,
            "capability": msg.capability,
        },
        "tags": [f"unit:{unit_id}", f"trace:{msg.trace_id[:8]}"],
        "run_name": run_name or f"{unit_id}.handle",
    }


def langsmith_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"} and bool(
        os.getenv("LANGSMITH_API_KEY")
    )
