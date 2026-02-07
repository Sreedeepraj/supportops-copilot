import hashlib
from typing import Dict, List

from app.core.settings import settings
from app.memory.store import init_db, add_message, get_recent_messages
from app.memory.vector import get_memory_store, add_memory_texts, search_memories

# Reuse the same persist dir you use for your Chroma doc store
PERSIST_DIR = "data/vector_store"


def ensure_memory_ready() -> None:
    init_db(settings.MEMORY_DB_PATH)


def _make_id(user_id: str, role: str, content: str) -> str:
    h = hashlib.sha256(f"{user_id}:{role}:{content}".encode("utf-8")).hexdigest()[:16]
    return f"{user_id}:{role}:{h}"


# -------------------------
# Short-term memory (session-scoped)
# -------------------------
def load_short_term(session_id: str) -> List[Dict]:
    return get_recent_messages(settings.MEMORY_DB_PATH, session_id, settings.MEMORY_MAX_MESSAGES)


# -------------------------
# Long-term memory (user-scoped)
# -------------------------
def load_long_term(user_id: str, question: str) -> List[Dict]:
    store = get_memory_store(PERSIST_DIR, settings.MEMORY_COLLECTION)
    return search_memories(
        store,
        query=question,
        top_k=settings.MEMORY_TOP_K,
        where={"user_id": user_id},
    )


# -------------------------
# Persist both stores
# -------------------------
def remember_turn(user_id: str, session_id: str, question: str, answer: str) -> None:
    # short-term: store raw messages per session
    add_message(settings.MEMORY_DB_PATH, session_id, "user", question)
    add_message(settings.MEMORY_DB_PATH, session_id, "assistant", answer)

    # long-term: store semantic memories per user
    store = get_memory_store(PERSIST_DIR, settings.MEMORY_COLLECTION)

    uid = _make_id(user_id, "user", question)
    aid = _make_id(user_id, "assistant", answer)

    texts = [question, answer]
    metadatas = [
        {"user_id": user_id, "session_id": session_id, "role": "user", "type": "chat"},
        {"user_id": user_id, "session_id": session_id, "role": "assistant", "type": "chat"},
    ]
    add_memory_texts(store, texts=texts, metadatas=metadatas, ids=[uid, aid])