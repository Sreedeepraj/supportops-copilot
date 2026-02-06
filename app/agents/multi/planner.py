import json
import logging
import time
from copy import deepcopy
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field, ValidationError

from app.llm.client import get_chat_model

logger = logging.getLogger(__name__)


# ---------- Schema (Pydantic) ----------

class PlanModel(BaseModel):
    steps: List[str] = Field(..., min_length=2, max_length=4)


# ---------- Stats helpers (same pattern as Day 6/7) ----------

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


# ---------- Parsing helpers ----------

def _strip_code_fences(text: str) -> str:
    """
    LLMs sometimes wrap JSON in ```json ... ``` fences.
    This removes a single fenced block if present.
    """
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        # remove first line (``` or ```json)
        if len(lines) >= 2:
            lines = lines[1:]
        # remove trailing ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _default_plan() -> List[str]:
    return ["Retrieve relevant docs", "Answer with citations", "Verify grounding"]


def _parse_plan_json(text: str) -> List[str]:
    """
    Parse JSON -> validate with Pydantic -> normalize & guardrails.
    Always returns a plan (falls back if invalid).
    """
    try:
        raw = _strip_code_fences(text)
        data = json.loads(raw)
        model = PlanModel.model_validate(data)  # pydantic v2
        steps = [s.strip() for s in model.steps if s and s.strip()]
        if len(steps) < 2:
            return _default_plan()
        return steps[:4]
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        return _default_plan()


# ---------- Node ----------

def node_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    start, stats = _with_timing(state, "plan")

    llm = get_chat_model()
    q = state["question"]

    schema_json = PlanModel.model_json_schema()

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a planning agent for a documentation QA assistant.\n"
                "Create a minimal plan of 2 to 4 steps.\n"
                "Rules:\n"
                "- Each step max 8 words.\n"
                "- Prefer: retrieve docs → answer → verify.\n"
                "- Return ONLY valid JSON (no extra text) that matches this schema:\n"
                f"{json.dumps(schema_json, indent=2)}\n"
                "\nExample output:\n"
                '{"steps":["Retrieve relevant docs","Answer with citations","Verify grounding"]}'
            ),
        },
        {"role": "user", "content": q},
    ]

    result = llm.invoke(prompt)
    plan = _parse_plan_json(result.content)

    usage = _extract_token_usage(result)
    if usage:
        stats["tokens"]["plan"] = usage

    stats = _finish_timing(stats, "plan", start)
    logger.info("multi_plan", extra={"steps": len(plan)})

    return {"plan": plan, "current_step": 0, "attempts": 0, "stats": stats}