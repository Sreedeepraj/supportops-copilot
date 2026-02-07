from typing import List, Dict, Optional
from langchain_chroma import Chroma
from app.ingest.embedder import get_embedder  


def get_memory_store(persist_dir: str, collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=get_embedder(),
    )


def add_memory_texts(store: Chroma, texts: List[str], metadatas: List[Dict], ids: Optional[List[str]] = None) -> None:
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def search_memories(store: Chroma, query: str, top_k: int, where: Optional[Dict] = None) -> List[Dict]:
    """
    Returns: [{"text": str, "metadata": dict, "score": float}, ...]
    """
    results = store.similarity_search_with_score(query, k=top_k, filter=where)
    out: List[Dict] = []
    for doc, score in results:
        out.append({"text": doc.page_content, "metadata": doc.metadata, "score": float(score)})
    return out