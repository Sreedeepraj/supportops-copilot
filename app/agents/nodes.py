import logging
from typing import Dict, Any
from app.retrieval.vector_store import get_vector_store
from app.retrieval.retriever import retrieve
from app.llm.client import get_chat_model
from app.rag.prompting import build_context, build_messages

logger = logging.getLogger(__name__)
PERSIST_DIR = "data/vector_store"

def node_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    store = get_vector_store(PERSIST_DIR)
    question = state["question"]
    top_k = state.get("top_k", 4)
    metadata_filter = state.get("metadata_filter")

    chunks = retrieve(store, question, top_k=top_k, metadata_filter=metadata_filter)
    logger.info("agent_retrieve", extra={"retrieved": len(chunks)})
    return {"chunks": chunks}

def _looks_relevant(question: str, chunks) -> bool:
    q = question.lower()
    joined = " ".join(c["text"].lower() for c in chunks)

    # pick a few “content words” cheaply
    keywords = []
    for w in ["agent", "agents", "tool", "tools", "langgraph", "planner", "executor"]:
        if w in q:
            keywords.append(w)

    if not keywords:
        # If no obvious keywords, fall back to “we have enough chunks”
        return len(chunks) >= 2

    return any(k in joined for k in keywords)

def node_assess(state: Dict[str, Any]) -> Dict[str, Any]:
    chunks = state.get("chunks", [])
    ok = _looks_relevant(state["question"], chunks)
    logger.info("agent_assess", extra={"retrieved": len(chunks), "ok": ok})
    return {"_retrieval_ok": ok}

def node_rewrite(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    One cheap rewrite call. Output should be a better search query, not the final answer.
    """
    llm = get_chat_model()

    question = state["question"]
    prompt = [
        {"role": "system", "content": "Rewrite the user question into a short search query for documentation retrieval. Return ONLY the rewritten query."},
        {"role": "user", "content": question},
    ]

    rewritten = llm.invoke(prompt).content.strip().strip('"')
    logger.info("agent_rewrite", extra={"rewritten": rewritten[:120]})
    return {"rewritten_question": rewritten}

def node_retrieve_rewritten(state: Dict[str, Any]) -> Dict[str, Any]:
    store = get_vector_store(PERSIST_DIR)
    q = state.get("rewritten_question") or state["question"]
    top_k = state.get("top_k", 4)
    metadata_filter = state.get("metadata_filter")

    chunks = retrieve(store, q, top_k=top_k, metadata_filter=metadata_filter)
    logger.info("agent_retrieve2", extra={"retrieved": len(chunks)})
    return {"chunks": chunks}

def node_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    chunks = state.get("chunks", [])
    if not chunks:
        return {
            "answer": "I don't know based on the provided documents.",
            "citations": [],
        }

    context = build_context(chunks)
    messages = build_messages(state["question"], context)

    llm = get_chat_model()
    result = llm.invoke(messages)

    citations = [
        {
            "id": c["metadata"].get("id"),
            "source": c["metadata"].get("source"),
            "score": c["score"],
        }
        for c in chunks
    ]

    return {"answer": result.content, "citations": citations}