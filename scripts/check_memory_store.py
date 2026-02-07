from app.core.settings import settings
from app.memory.vector import get_memory_store
from dotenv import load_dotenv
load_dotenv()

PERSIST_DIR = "data/vector_store"

def main():
    store = get_memory_store(PERSIST_DIR, settings.MEMORY_COLLECTION)

    # Chroma exposes underlying collection
    col = store._collection
    print("collection:", col.name)
    print("count:", col.count())

    # peek a few items
    peek = col.peek(5)
    print("peek ids:", peek.get("ids"))
    print("peek metas:", peek.get("metadatas"))
    print("peek docs:", [d[:80] for d in (peek.get("documents") or [])])

if __name__ == "__main__":
    main()