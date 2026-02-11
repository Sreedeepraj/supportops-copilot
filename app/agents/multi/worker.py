from __future__ import annotations
import logging
import time
from typing import Dict, Any, Optional
from copy import deepcopy
from app.memory.service import load_short_term
from app.agents.graph import build_graph as build_rag_agent_graph
from app.mcp.mcp_client import MCPTools
from app.guardrails.budgets import clamp_top_k, trim_chunks_by_budget, trim_memory_lines
from app.guardrails.injection import sanitize_chunks
from app.guardrails.grounding import should_abstain

logger = logging.getLogger(__name__)
_rag_graph = build_rag_agent_graph()
mcp_tools = MCPTools()


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

    session_id = state.get("session_id", "default-session")
    user_id = state.get("user_id", "default-user")

    top_k = clamp_top_k(int(state.get("top_k", 4) or 4), max_k=4)

    # Memory retrieval (short-term local, long-term via MCP tool)
    short_mem = load_short_term(session_id)
    long_mem = mcp_tools.memory_search(user_id, state["question"], top_k=3)

    stats.setdefault("memory", {})
    stats["memory"]["short_count"] = len(short_mem) if isinstance(short_mem, list) else 0
    stats["memory"]["long_count"] = len(long_mem) if isinstance(long_mem, list) else 0

    # Build memory lines, then ✅ Guardrail: trim memory injected
    memory_lines: list[str] = []
    if short_mem:
        memory_lines.append("SHORT-TERM CHAT HISTORY (most recent):")
        for m in short_mem:
            # load_short_term returns {"role": ..., "content": ...}
            memory_lines.append(f'{m.get("role", "user").upper()}: {m.get("content", "")}')

    if long_mem:
        memory_lines.append("LONG-TERM RELEVANT MEMORIES:")
        for i, mm in enumerate(long_mem, 1):
            # canonical: {"text": ..., "metadata": ..., "score": ...}
            text = mm.get("text") if isinstance(mm, dict) else getattr(mm, "text", str(mm))
            memory_lines.append(f"[M{i}] {text}")

    # Trim lines + line length (hard prompt budget)
    memory_lines = trim_memory_lines(memory_lines, max_lines=10, max_line_chars=200)
    memory_block = "\n".join(memory_lines).strip()

    # Augment question with memory
    augmented_question = state["question"]
    if memory_block:
        augmented_question = (
            f"{state['question']}\n\n---\n"
            f"MEMORY CONTEXT (use only if relevant):\n{memory_block}\n---"
        )

    # Call sub-agent RAG
    init_state = {
        "question": augmented_question,
        "top_k": top_k,
        "metadata_filter": state.get("metadata_filter"),
        "path": "direct",
        "stats": {"steps": [], "latency_ms": {}, "tokens": {}},
    }

    out = _rag_graph.invoke(init_state)

    # merge rag stats inside multi stats (keep separate namespaces to avoid collisions)
    rag_stats = out.get("stats") or {}
    stats.setdefault("sub", {})
    stats["sub"]["rag"] = rag_stats

    # Guardrails on RAG outputs (deterministic + safe)
    chunks = out.get("chunks") or []
    answer = out.get("answer", "I don't know.")
    citations = out.get("citations", [])

    # 1) sanitize retrieved text (prompt-injection defense)
    chunks = sanitize_chunks(chunks)

    # 2) trim docs context to a hard budget
    chunks = trim_chunks_by_budget(chunks, max_total_chars=6000)

    # 3) grounding check: if answer not supported by (memory+docs), abstain
    docs_only_context = "\n\n".join(
    [(c.get("text") or "").strip() for c in chunks if (c.get("text") or "").strip()])
    abstain, gscore = should_abstain(answer=answer, context=docs_only_context, threshold=0.12)

    stats.setdefault("guardrails", {})
    stats["guardrails"]["top_k"] = top_k
    stats["guardrails"]["memory_lines_used"] = len(memory_lines)
    stats["guardrails"]["chunks_used"] = len(chunks)
    stats["guardrails"]["grounding_score"] = gscore
    stats["guardrails"]["abstained"] = bool(abstain)

    if abstain and len(chunks) == 0:
        answer = "I don't know based on the provided sources."
        stats["guardrails"]["abstained"] = True
    else:
        stats["guardrails"]["abstained"] = abstain

    stats = _finish_timing(stats, "work", start)
    logger.info("multi_work_done", extra={"attempts": attempts, "path": out.get("path")})

    return {
        "attempts": attempts,
        "answer": answer,
        "citations": citations,
        "retrieved": len(chunks),
        "path": out.get("path", "direct"),
        "rewritten_question": out.get("rewritten_question"),
        "stats": stats,
    }