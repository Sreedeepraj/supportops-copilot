from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.rag.service import answer_question

router = APIRouter()

class QARequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(4, ge=1, le=10)

class QAResponse(BaseModel):
    answer: str
    citations: list[dict]
    stats: dict

@router.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    # Enforce semantic chunks (since you built both strategies)
    return answer_question(
        question=req.question,
        top_k=req.top_k,
        metadata_filter={"chunk_strategy": "semantic"},
    )