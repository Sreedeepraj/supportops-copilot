import logging
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def get_vector_store(persist_dir: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    store = Chroma(
        collection_name="docs",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    logger.info(f"Vector store initialized at {persist_dir}")
    return store


def add_records(store, records: List[Dict[str, Any]]):
    texts = [r["text"] for r in records]
    metadatas = [
        {
            "id": r["id"],
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],
            "chunk_strategy": r["chunk_strategy"],
            "source": r["source"],
            "ingested_at": r["ingested_at"],
        }
        for r in records
    ]
    ids = [r["id"] for r in records]

    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    logger.info(f"Added {len(records)} records to vector store")