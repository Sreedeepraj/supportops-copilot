from dotenv import load_dotenv
load_dotenv()

from app.ingest.pipeline import run_ingestion
from app.retrieval.vector_store import get_vector_store, add_records

if __name__ == "__main__":
    records = run_ingestion(
        "data/docs/langchain/langchain",
        max_docs=10,
        chunk_strategy="semantic",
    )

    store = get_vector_store("data/vector_store")
    add_records(store, records)