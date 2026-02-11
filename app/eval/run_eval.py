from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from app.eval.types import EvalCase, EvalResult
from app.api.qa_multi_agent import graph  # uses the graph you already expose in API
from dotenv import load_dotenv
load_dotenv()

CASES_PATH = Path("data/eval/cases.jsonl")
REPORT_PATH = Path("data/eval/report.json")


def _load_cases(path: Path) -> List[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Eval cases not found: {path}")

    cases: List[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        cases.append(
            EvalCase(
                id=str(obj["id"]),
                user_id=str(obj.get("user_id", "u_eval")),
                session_id=str(obj.get("session_id", "s_eval")),
                question=str(obj["question"]),
                expect_abstain=bool(obj.get("expect_abstain", False)),
                expect_contains=list(obj.get("expect_contains", [])),
                notes=obj.get("notes"),
            )
        )
    return cases


def _normalize_answer(a: str) -> str:
    return (a or "").strip()


def _is_abstain(answer: str) -> bool:
    # align with your guardrail behavior
    a = _normalize_answer(answer).lower()
    return a in {
        "i don't know.",
        "i dont know.",
        "i don't know based on the provided sources.",
        "i dont know based on the provided sources.",
    } or a.startswith("i don't know based on")


def _contains_all(answer: str, required: List[str]) -> bool:
    a = _normalize_answer(answer).lower()
    return all(r.lower() in a for r in required)


def _eval_one(case: EvalCase) -> EvalResult:
    init_state: Dict[str, Any] = {
        "user_id": case.user_id,
        "session_id": case.session_id,
        "question": case.question,
        "top_k": 4,
        "metadata_filter": None,
        "attempts": 0,
        "done": False,
        "stats": {"steps": [], "latency_ms": {}, "tokens": {}},
    }

    t0 = time.perf_counter()
    out = graph.invoke(init_state)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    answer = out.get("answer", "")
    abstained = _is_abstain(answer)

    stats = out.get("stats") or {}
    guard = (stats.get("guardrails") or {})
    grounding_score = guard.get("grounding_score")

    retrieved = int(out.get("retrieved", 0) or 0)

    fail_reasons: List[str] = []

    # 1) abstain expectation
    if case.expect_abstain and not abstained:
        fail_reasons.append("expected_abstain_but_answered")
    if (not case.expect_abstain) and abstained:
        fail_reasons.append("expected_answer_but_abstained")

    # 2) contains expectation (only if we expected a real answer)
    if (not case.expect_abstain) and case.expect_contains:
        if not _contains_all(answer, case.expect_contains):
            fail_reasons.append(f"missing_expected_terms:{case.expect_contains}")

    # 3) optional sanity checks
    if (not case.expect_abstain) and retrieved <= 0:
        fail_reasons.append("no_retrieval_for_answerable_case")

    # record timing
    stats.setdefault("eval", {})
    stats["eval"]["elapsed_ms"] = elapsed_ms

    passed = len(fail_reasons) == 0

    return EvalResult(
        case_id=case.id,
        passed=passed,
        abstained=abstained,
        answer=_normalize_answer(answer),
        retrieved=retrieved,
        grounding_score=grounding_score if grounding_score is None else float(grounding_score),
        fail_reasons=fail_reasons,
        stats=stats,
    )


def main() -> int:
    cases = _load_cases(CASES_PATH)
    results: List[EvalResult] = []

    print(f"Loaded {len(cases)} eval cases from {CASES_PATH}")
    print("-" * 80)

    passed = 0
    for c in cases:
        r = _eval_one(c)
        results.append(r)

        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {c.id}: {c.question}")
        if not r.passed:
            sub = (r.stats.get("sub") or {}).get("rag") or {}
            print(f"  grounding={r.grounding_score} retrieved={r.retrieved}")
        else:
            print(f"  abstained={r.abstained} retrieved={r.retrieved} grounding={r.grounding_score}")

        if r.passed:
            passed += 1

    print("-" * 80)
    print(f"Summary: {passed}/{len(results)} passed")

    # write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps([r.__dict__ for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"Saved report to {REPORT_PATH}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())