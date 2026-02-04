from typing import List, Dict, Any

def build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Create a context block with citations.
    We include chunk id + source so the model can cite.
    """
    parts = []
    for c in chunks:
        meta = c["metadata"]
        chunk_id = meta.get("id") or f"{meta.get('doc_id')}:{meta.get('chunk_id')}"
        source = meta.get("source", "unknown")

        parts.append(
            f"[{chunk_id}] source={source}\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def build_messages(question: str, context: str) -> list[dict]:
    system = (
        "You are a careful assistant. Answer the question using ONLY the provided context.\n"
        "If the context is insufficient, say you don't know.\n"
        "When you use facts from the context, cite them using the bracketed chunk ids like [doc:chunk]."
    )

    user = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Return a clear answer with citations."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]