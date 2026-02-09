from typing import TypedDict, List, Dict, Any, Optional


class MultiAgentState(TypedDict, total=False):
    user_id: str
    session_id: str
    # input
    question: str
    top_k: int
    metadata_filter: Optional[dict]

    # planner output
    plan: List[str]                 # e.g. ["Retrieve docs", "Answer", "Verify grounding"]
    current_step: int               # index into plan

    # worker outputs
    answer: str
    citations: List[Dict[str, Any]]
    retrieved: int
    rewritten_question: str
    path: str                       # "direct" | "rewrite"

    # critic outputs
    critique: str
    done: bool

    # loop control
    attempts: int                   # number of worker attempts (for retry limit)

    # observability
    stats: Dict[str, Any]