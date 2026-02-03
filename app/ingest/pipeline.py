import logging
from app.ingest.loader import load_markdown_files
from app.ingest.chunker import fixed_chunk, semantic_chunk
from app.ingest.embedder import get_embedder
from app.core.retry import retry_on_transient_failure
from app.core.retryable import RetryableError
import hashlib
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def _doc_id_from_source(source: str) -> str:
    # Stable ID derived from source path
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


@retry_on_transient_failure()
def embed_chunks(embedder, chunks: list[str]) -> list[list[float]]:
    try:
        return embedder.embed_documents(chunks)
    except Exception as e:
        # Mark as retryable for now (we'll narrow this later to 429/5xx/timeouts)
        raise RetryableError(str(e)) from e


def run_ingestion(root_dir: str, max_docs: int = 20, chunk_strategy: str = "fixed") -> list[dict]:
    """
    Ingest a limited number of docs first to control cost.
    We'll scale up after everything is stable.
    """
    docs = load_markdown_files(root_dir)
    docs = docs[:max_docs]
    logger.info(f"Ingesting {len(docs)} documents (max_docs={max_docs})")

    embedder = get_embedder()
    records = []

    for doc in docs:
        if chunk_strategy == "fixed":
            chunks = fixed_chunk(doc["text"])
        elif chunk_strategy == "semantic":
            chunks = semantic_chunk(doc["text"])
        else:
            raise ValueError(f"Unknown chunk_strategy: {chunk_strategy}")
        
        vectors = embed_chunks(embedder, chunks)

        doc_id = _doc_id_from_source(doc["source"])

        logger.info(f"[{chunk_strategy}] doc_id={doc_id} chunks={len(chunks)} source={doc['source']}")

        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = idx
            chunk_uid = f"{doc_id}:{chunk_id}"
            records.append(
                {
                    "id": chunk_uid,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_strategy": chunk_strategy,
                    "source": doc["source"],
                    "text": chunk,
                    "vector": vector,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    logger.info(f"Ingestion complete. Embedded chunks={len(records)}")
    return records