import logging
import time
from typing import Dict, Any, Optional
from copy import deepcopy

from app.llm.client import get_chat_model

logger = logging.getLogger(__name__)


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


def _extract_token_usage(result) -> Optional[dict]:
    rm = getattr(result, "response_metadata", None)
    if isinstance(rm, dict):
        for k in ("token_usage", "usage"):
            if k in rm and isinstance(rm[k], dict):
                return rm[k]
    um = getattr(result, "usage_metadata", None)
    if isinstance(um, dict):
        return um
    return None


def node_critic(state: Dict[str, Any]) -> Dict[str, Any]:
    start, stats = _with_timing(state, "critic")

    answer = (state.get("answer") or "").strip()
    citations = state.get("citations") or []

    # Safe accept: explicitly unknown
    if answer.lower().startswith("i don't know"):
        stats = _finish_timing(stats, "critic", start)
        return {"done": True, "critique": "Answer is safely abstaining.", "stats": stats}

    # Hard rule: needs citations for non-abstain answers
    if len(citations) == 0:
        stats = _finish_timing(stats, "critic", start)
        return {"done": False, "critique": "Missing citations; retry with rewrite.", "stats": stats}

    llm = get_chat_model()
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a strict grounding critic.\n"
                "Given an answer and citation list, decide if the answer is grounded.\n"
                "Return ONLY one token: PASS or FAIL.\n"
            ),
        },
        {
            "role": "user",
            "content": f"ANSWER:\n{answer}\n\nCITATIONS:\n{citations}",
        },
    ]

    result = llm.invoke(prompt)
    verdict = result.content.strip().upper()

    usage = _extract_token_usage(result)
    if usage:
        stats["tokens"]["critic"] = usage

    done = verdict == "PASS"
    critique = "Grounded answer." if done else "Potentially ungrounded; retry once."

    stats = _finish_timing(stats, "critic", start)
    logger.info("multi_critic", extra={"verdict": verdict})

    return {"done": done, "critique": critique, "stats": stats}