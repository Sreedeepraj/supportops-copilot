from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agents.graph import build_graph

router = APIRouter()
graph = build_graph()

class QARequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(4, ge=1, le=10)

@router.post("/qa-agent")
def qa_agent(req: QARequest):
    state = {
        "question": req.question,
        "top_k": req.top_k,
        "metadata_filter": {"chunk_strategy": "semantic"},
    }

    out = graph.invoke(state)

    # Detect which path we took
    path = "rewrite" if out.get("rewritten_question") else "direct"

    return {
        "answer": out.get("answer", "I don't know."),
        "citations": out.get("citations", []),
        "path": path,
        "rewritten_question": out.get("rewritten_question"),
        "retrieved": len(out.get("chunks", [])),
    }