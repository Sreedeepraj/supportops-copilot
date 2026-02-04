import logging
import time
from typing import Dict, Any, Optional

from app.retrieval.vector_store import get_vector_store
from app.retrieval.retriever import retrieve
from app.llm.client import get_chat_model
from app.rag.prompting import build_context, build_messages

logger = logging.getLogger(__name__)

PERSIST_DIR = "data/vector_store"


def answer_question(question: str, top_k: int = 4, metadata_filter: Optional[dict] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()

    store = get_vector_store(PERSIST_DIR)

    t1 = time.perf_counter()
    chunks = retrieve(
        store,
        question,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )
    t2 = time.perf_counter()

    # If retrieval is empty, fail gracefully
    if not chunks:
        return {
            "answer": "I don't know based on the provided documents.",
            "citations": [],
            "stats": {
                "retrieved": 0,
                "latency_ms": {
                    "vector_store_init": int((t1 - t0) * 1000),
                    "retrieval": int((t2 - t1) * 1000),
                    "llm": 0,
                    "total": int((time.perf_counter() - t0) * 1000),
                },
                "tokens": {},
            },
        }

    context = build_context(chunks)
    messages = build_messages(question, context)

    llm = get_chat_model()
    t3 = time.perf_counter()

    # LangChain returns an AIMessage; usage metadata depends on provider/version.
    result = llm.invoke(messages)

    t4 = time.perf_counter()

    # Extract token usage if present
    usage = {}
    try:
        # Newer LangChain/OpenAI often attaches usage in response_metadata
        usage = (result.response_metadata or {}).get("token_usage", {}) or {}
    except Exception:
        usage = {}

    citations = [
        {
            "id": c["metadata"].get("id"),
            "source": c["metadata"].get("source"),
            "score": c["score"],
        }
        for c in chunks
    ]

    stats = {
        "retrieved": len(chunks),
        "latency_ms": {
            "vector_store_init": int((t1 - t0) * 1000),
            "retrieval": int((t2 - t1) * 1000),
            "llm": int((t4 - t3) * 1000),
            "total": int((t4 - t0) * 1000),
        },
        "tokens": usage,
    }

    logger.info(
        "rag_answer_completed",
        extra={
            "retrieved": stats["retrieved"],
            "latency_ms": stats["latency_ms"],
            "tokens": stats["tokens"],
        },
    )

    return {
        "answer": result.content,
        "citations": citations,
        "stats": stats,
    }