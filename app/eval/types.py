from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalCase:
    id: str
    user_id: str
    session_id: str
    question: str
    expect_abstain: bool
    expect_contains: List[str]
    notes: Optional[str] = None


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    abstained: bool
    answer: str
    retrieved: int
    grounding_score: float | None
    fail_reasons: List[str]
    stats: dict