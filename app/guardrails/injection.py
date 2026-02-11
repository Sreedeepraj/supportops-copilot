import re
from typing import List, Dict, Any

_INJECTION_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"system prompt",
    r"developer message",
    r"you are chatgpt",
    r"do not follow",
    r"tool call",
    r"function call",
]

_RX = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

def sanitize_context_text(text: str) -> str:
    # Remove lines that look like prompt injection attempts
    safe_lines = []
    for ln in text.splitlines():
        if _RX.search(ln):
            continue
        safe_lines.append(ln)
    return "\n".join(safe_lines).strip()

def sanitize_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in chunks:
        cc = dict(c)
        cc["text"] = sanitize_context_text(c.get("text", "") or "")
        out.append(cc)
    return out