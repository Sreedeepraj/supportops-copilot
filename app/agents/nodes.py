import logging
import time
from copy import deepcopy
from typing import Dict, Any, List, Optional, Tuple

from app.retrieval.vector_store import get_vector_store
from app.retrieval.retriever import retrieve
from app.llm.client import get_chat_model
from app.rag.prompting import build_context, build_messages

logger = logging.getLogger(__name__)
PERSIST_DIR = "data/vector_store"


def _init_stats(existing: Optional[dict]) -> dict:
    """
    Ensure stats has expected keys. Uses deepcopy to avoid mutating state in-place.
    """
    base = {"steps": [], "latency_ms": {}, "tokens": {}}
    if not existing:
        return base
    out = deepcopy(existing)
    out.setdefault("steps", [])
    out.setdefault("latency_ms", {})
    out.setdefault("tokens", {})
    return out


def _with_timing(state: dict, step: str) -> Tuple[float, dict]:
    start = time.perf_counter()
    stats = _init_stats(state.get("stats"))
    stats["steps"].append(step)
    return start, stats


def _finish_timing(stats: dict, step: str, start: float) -> dict:
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    stats["latency_ms"][step] = elapsed_ms
    return stats


def _extract_token_usage(result) -> Optional[dict]:
    """
    Best-effort extraction across common LangChain response shapes.
    Returns None if usage is not available.
    """
    rm = getattr(result, "response_metadata", None)
    if isinstance(rm, dict):
        for k in ("token_usage", "usage"):
            if k in rm and isinstance(rm[k], dict):
                return rm[k]

    um = getattr(result, "usage_metadata", None)
    if isinstance(um, dict):
        return um

    return None


def _keyword_signal(question: str, chunks: List[Dict[str, Any]]) -> bool:
    """
    Simple heuristic for Day 6:
    - If we can find key terms from the question in retrieved text, treat it as ok.
    Avoid relying on score thresholds (corpus-dependent).
    """
    if not chunks:
        return False

    q = question.lower()
    joined = " ".join(c.get("text", "").lower() for c in chunks)

    candidates = [
        "agent",
        "agents",
        "tool",
        "tools",
        "langgraph",
        "planner",
        "executor",
        "retrieval",
        "rag",
    ]
    terms = [t for t in candidates if t in q]

    if not terms:
        return len(chunks) >= 2

    return any(t in joined for t in terms)


def node_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    start, stats = _with_timing(state, "retrieve")

    store = get_vector_store(PERSIST_DIR)
    chunks = retrieve(
        store,
        state["question"],
        top_k=state.get("top_k", 4),
        metadata_filter=state.get("metadata_filter"),
    )

    stats = _finish_timing(stats, "retrieve", start)
    logger.info("agent_retrieve", extra={"retrieved": len(chunks)})

    return {"chunks": chunks, "stats": stats}


def node_assess(state: Dict[str, Any]) -> Dict[str, Any]:
    start, stats = _with_timing(state, "assess")

    chunks = state.get("chunks", [])
    ok = _keyword_signal(state["question"], chunks)

    stats = _finish_timing(stats, "assess", start)
    logger.info("agent_assess", extra={"retrieved": len(chunks), "ok": ok})

    return {"retrieval_ok": ok, "stats": stats}


def node_rewrite(state: Dict[str, Any]) -> Dict[str, Any]:
    start, stats = _with_timing(state, "rewrite")

    llm = get_chat_model()
    q = state["question"]

    prompt = [
        {
            "role": "system",
            "content": "Rewrite the question into a short search query for documentation retrieval. Return ONLY the rewritten query.",
        },
        {"role": "user", "content": q},
    ]

    result = llm.invoke(prompt)
    rewritten = result.content.strip().strip('"')

    usage = _extract_token_usage(result)
    if usage:
        stats["tokens"]["rewrite"] = usage

    stats = _finish_timing(stats, "rewrite", start)
    logger.info("agent_rewrite", extra={"rewritten": rewritten[:120]})

    return {"rewritten_question": rewritten, "path": "rewrite", "stats": stats}


def node_retrieve_rewritten(state: Dict[str, Any]) -> Dict[str, Any]:
    start, stats = _with_timing(state, "retrieve2")

    store = get_vector_store(PERSIST_DIR)
    q2 = state.get("rewritten_question") or state["question"]

    chunks = retrieve(
        store,
        q2,
        top_k=state.get("top_k", 4),
        metadata_filter=state.get("metadata_filter"),
    )

    stats = _finish_timing(stats, "retrieve2", start)
    logger.info("agent_retrieve2", extra={"retrieved": len(chunks)})

    return {"chunks": chunks, "stats": stats}


def node_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    start, stats = _with_timing(state, "answer")

    chunks = state.get("chunks", [])
    if not chunks:
        stats = _finish_timing(stats, "answer", start)
        return {
            "answer": "I don't know based on the provided documents.",
            "citations": [],
            "stats": stats,
        }

    context = build_context(chunks)
    messages = build_messages(state["question"], context)

    llm = get_chat_model()
    result = llm.invoke(messages)

    usage = _extract_token_usage(result)
    if usage:
        stats["tokens"]["answer"] = usage

    citations = [
        {
            "id": c.get("metadata", {}).get("id"),
            "source": c.get("metadata", {}).get("source"),
            "score": c.get("score"),
        }
        for c in chunks
    ]

    stats = _finish_timing(stats, "answer", start)

    return {"answer": result.content, "citations": citations, "stats": stats}