import time
from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.memory.service import remember_turn
from app.agents.multi.graph import build_multi_agent_graph

router = APIRouter()
graph = build_multi_agent_graph()


class QAMultiRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=3)
    top_k: int = Field(4, ge=1, le=10)


@router.post("/qa-multi")
def qa_multi(req: QAMultiRequest):
    init_state = {
        "user_id": req.user_id,
        "session_id": req.session_id,
        "question": req.question,
        "top_k": req.top_k,
        "metadata_filter": {"chunk_strategy": "semantic"},
        "stats": {"steps": [], "latency_ms": {}, "tokens": {}},
    }

    t0 = time.perf_counter()
    out = graph.invoke(init_state)
    remember_turn(
    user_id=req.user_id,
    session_id=req.session_id,
    question=req.question,
    answer=out.get("answer", ""),
    )
    total_ms = int((time.perf_counter() - t0) * 1000)

    stats = out.get("stats") or {"steps": [], "latency_ms": {}, "tokens": {}}
    stats.setdefault("latency_ms", {})
    stats["latency_ms"]["total"] = total_ms

    return {
        "answer": out.get("answer", "I don't know."),
        "citations": out.get("citations", []),
        "plan": out.get("plan", []),
        "attempts": out.get("attempts", 0),
        "critique": out.get("critique"),
        "done": out.get("done"),
        "stats": stats,
    }