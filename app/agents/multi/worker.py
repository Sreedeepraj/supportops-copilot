import logging
import time
from typing import Dict, Any, Optional
from copy import deepcopy

from app.agents.graph import build_graph as build_rag_agent_graph

logger = logging.getLogger(__name__)
_rag_graph = build_rag_agent_graph()


def _init_stats(existing: Optional[dict]) -> dict:
    base = {"steps": [], "latency_ms": {}, "tokens": {}}
    if not existing:
        return base
    out = deepcopy(existing)
    out.setdefault("steps", [])
    out.setdefault("latency_ms", {})
    out.setdefault("tokens", {})
    return out


def _with_timing(state: dict, step: str):
    start = time.perf_counter()
    stats = _init_stats(state.get("stats"))
    stats["steps"].append(step)
    return start, stats


def _finish_timing(stats: dict, step: str, start: float) -> dict:
    stats["latency_ms"][step] = int((time.perf_counter() - start) * 1000)
    return stats


def node_work(state: Dict[str, Any]) -> Dict[str, Any]:
    start, stats = _with_timing(state, "work")
    attempts = int(state.get("attempts", 0)) + 1

    init_state = {
        "question": state["question"],
        "top_k": state.get("top_k", 4),
        "metadata_filter": state.get("metadata_filter"),
        "path": "direct",
        "stats": {"steps": [], "latency_ms": {}, "tokens": {}},
    }

    out = _rag_graph.invoke(init_state)

    # merge rag stats inside multi stats (keep separate namespaces to avoid collisions)
    rag_stats = out.get("stats") or {}
    stats.setdefault("sub", {})
    stats["sub"]["rag"] = rag_stats

    stats = _finish_timing(stats, "work", start)
    logger.info("multi_work_done", extra={"attempts": attempts, "path": out.get("path")})

    return {
        "attempts": attempts,
        "answer": out.get("answer", "I don't know."),
        "citations": out.get("citations", []),
        "retrieved": len(out.get("chunks", []) or []),
        "path": out.get("path", "direct"),
        "rewritten_question": out.get("rewritten_question"),
        "stats": stats,
    }