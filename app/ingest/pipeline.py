import logging
from app.ingest.loader import load_markdown_files
from app.ingest.chunker import fixed_chunk, semantic_chunk
from app.ingest.embedder import get_embedder
from app.core.retry import retry_on_transient_failure
from app.core.retryable import RetryableError

logger = logging.getLogger(__name__)


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
        

        logger.info(f"[{chunk_strategy}] Chunked source={doc['source']} chunks={len(chunks)}")

        vectors = embed_chunks(embedder, chunks)

        for chunk, vector in zip(chunks, vectors):
            records.append(
                {
                    "text": chunk,
                    "vector": vector,
                    "source": doc["source"],
                }
            )

    logger.info(f"Ingestion complete. Embedded chunks={len(records)}")
    return records