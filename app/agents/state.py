from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    question: str
    top_k: int
    metadata_filter: Optional[dict]

    rewritten_question: str
    chunks: List[Dict[str, Any]]
    retrieval_ok: bool

    answer: str
    citations: List[Dict[str, Any]]
    path: str  # "direct" | "rewrite"

    stats: Dict[str, Any]  # {"steps": [...], "latency_ms": {...}, "tokens": {...}}