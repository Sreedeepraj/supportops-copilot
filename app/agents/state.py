from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    question: str
    rewritten_question: str
    top_k: int
    metadata_filter: Optional[dict]

    chunks: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]

    path: str  # "direct" or "rewrite"