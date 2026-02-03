import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def retrieve(store,query: str,top_k: int = 4,metadata_filter: Optional[dict] = None,dedupe_by_source: bool = True) -> List[Dict[str, Any]]:
    """
    Retrieves top_k chunks using similarity search.
    - metadata_filter is enforced at the vector DB layer when supported.
    - dedupe_by_source prevents multiple chunks from the same file dominating results.
    """

    k_fetch = top_k * 6  # fetch extra then dedupe

    # Chroma/LangChain versions vary: some use `filter=`, some `where=`.
    # We'll try `filter` first, then fall back to `where`.
    try:
        results = store.similarity_search_with_score(query,k=k_fetch,filter=metadata_filter)
    except TypeError:
        results = store.similarity_search_with_score(query,k=k_fetch,where=metadata_filter)

    chunks: List[Dict[str, Any]] = []
    seen_sources = set()

    for doc, score in results:
        source = (doc.metadata.get("source") or "").strip()

        if dedupe_by_source and source:
            if source in seen_sources:
                continue
            seen_sources.add(source)

        chunks.append(
            {
                "text": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
            }
        )

        if len(chunks) >= top_k:
            break

    logger.info(
        f"Retrieved chunks={len(chunks)} top_k={top_k} "
        f"filter={metadata_filter} query='{query[:60]}'"
    )
    return chunks