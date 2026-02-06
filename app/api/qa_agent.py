import time
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agents.graph import build_graph

router = APIRouter()
graph = build_graph()


class QAAgentRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(4, ge=1, le=10)


@router.post("/qa-agent")
def qa_agent(req: QAAgentRequest):
    init_state = {
        "question": req.question,
        "top_k": req.top_k,
        "metadata_filter": {"chunk_strategy": "semantic"},
        "path": "direct",  # default; rewrite node flips to "rewrite"
        "stats": {"steps": [], "latency_ms": {}, "tokens": {}},
    }

    t0 = time.perf_counter()
    out = graph.invoke(init_state)
    total_ms = int((time.perf_counter() - t0) * 1000)

    stats = out.get("stats") or {"steps": [], "latency_ms": {}, "tokens": {}}
    stats.setdefault("latency_ms", {})
    stats["latency_ms"]["total"] = total_ms

    return {
        "answer": out.get("answer", "I don't know."),
        "citations": out.get("citations", []),
        "path": out.get("path", "direct"),
        "rewritten_question": out.get("rewritten_question"),
        "retrieved": len(out.get("chunks", []) or []),
        "stats": stats,
    }