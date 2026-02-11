from __future__ import annotations

import re
from typing import Tuple

_WORD = re.compile(r"[a-zA-Z0-9]+")

_ABSTAIN_PREFIXES = (
    "i don't know",
    "i dont know",
    "i don't know based on",
    "i dont know based on",
)

_STOPWORDS = {
    # keep it small—enough to avoid silly overlaps
    "i", "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "is", "are", "was", "were",
    "do", "don", "t", "not", "know",  # important: removes overlap from "I don't know"
}

def is_abstain_answer(answer: str) -> bool:
    a = (answer or "").strip().lower()
    return any(a.startswith(p) for p in _ABSTAIN_PREFIXES)

def _tokens(s: str) -> set[str]:
    toks = {m.group(0).lower() for m in _WORD.finditer(s or "")}
    return {t for t in toks if t not in _STOPWORDS}

def grounding_score(answer: str, context: str) -> float:
    a = _tokens(answer)
    c = _tokens(context)
    if not a:
        return 0.0
    return len(a & c) / max(1, len(a))

def should_abstain(answer: str, context: str, threshold: float = 0.12) -> Tuple[bool, float]:
    # If answer is already an abstain, respect it.
    if is_abstain_answer(answer):
        return True, 0.0

    score = grounding_score(answer, context)
    return (score < threshold, score)