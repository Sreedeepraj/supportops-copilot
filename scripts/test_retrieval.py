from dotenv import load_dotenv
load_dotenv()

from app.retrieval.vector_store import get_vector_store
from app.retrieval.retriever import retrieve

if __name__ == "__main__":
    store = get_vector_store("data/vector_store")

    query = "How do agents work in LangChain?"
    chunks = retrieve(store, query, top_k=3, metadata_filter={"chunk_strategy": "semantic"})

    for i, c in enumerate(chunks, 1):
        print(f"\n--- RESULT {i} ---")
        print("score:", c["score"])
        print("source:", c["metadata"]["source"])
        print(c["text"][:300])