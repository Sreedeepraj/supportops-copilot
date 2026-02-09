from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from app.core.settings import settings
from app.memory.service import ensure_memory_ready
from app.memory.service import remember_turn as remember_turn_local
from app.memory.service import load_long_term as load_long_term_local
from app.retrieval.vector_store import get_vector_store  # <-- adjust to your actual import
from app.retrieval.retriever import retrieve              # <-- adjust to your actual import

PERSIST_DIR = "data/vector_store"

mcp = FastMCP("SupportOps MCP")

@mcp.tool()
def memory_search(user_id: str, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Semantic search long-term memory for a user."""
    ensure_memory_ready()
    results = load_long_term_local(user_id, query)[:top_k]
    return [{"text": x["text"], "metadata": x.get("metadata", {}), "score": x.get("score")} for x in results]

@mcp.tool()
def memory_add(user_id: str, session_id: str, question: str, answer: str) -> Dict[str, Any]:
    """Persist a turn to short-term (SQLite) + long-term (Chroma) memory."""
    ensure_memory_ready()
    remember_turn_local(user_id=user_id, session_id=session_id, question=question, answer=answer)
    return {"status": "ok"}

@mcp.tool()
def docs_search(query: str, top_k: int = 4, metadata_filter: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Search your docs vector store (Chroma) and return chunks with metadata+score."""
    store = get_vector_store(PERSIST_DIR)
    chunks = retrieve(store, query, top_k=top_k, metadata_filter=metadata_filter)
    return chunks

if __name__ == "__main__":
    # Run MCP over streamable HTTP so your FastAPI app can call it.
    # FastMCP CLI also works, but this is simplest.
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8765)
