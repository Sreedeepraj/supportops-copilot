from __future__ import annotations
from typing import Any, Dict, List

def clamp_top_k(top_k: int, max_k: int = 4) -> int:
    try:
        k = int(top_k)
    except Exception:
        return max_k
    return max(1, min(k, max_k))

def trim_chunks_by_budget(chunks, max_total_chars=6000, min_chunks=2):
    total = 0
    kept = []
    for c in chunks:
        txt = c.get("text") or ""
        n = len(txt)
        if kept and total + n > max_total_chars and len(kept) >= min_chunks:
            break
        kept.append(c)
        total += n
    return kept

def trim_memory_lines(
    lines: List[str],
    max_lines: int = 6,
    max_line_chars: int = 200,
) -> List[str]:
    out: List[str] = []
    for ln in lines[:max_lines]:
        ln = ln.strip()
        if len(ln) > max_line_chars:
            ln = ln[:max_line_chars] + "…"
        out.append(ln)
    return out