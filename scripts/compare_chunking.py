from dotenv import load_dotenv
load_dotenv()

import time
from app.ingest.pipeline import run_ingestion


def run(strategy: str):
    start = time.perf_counter()
    records = run_ingestion("data/docs/langchain/langchain", max_docs=10, chunk_strategy=strategy)
    elapsed = time.perf_counter() - start

    lengths = [len(r["text"]) for r in records]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    print(f"\n--- {strategy.upper()} ---")
    print(f"chunks: {len(records)}")
    print(f"avg_chunk_chars: {avg_len:.1f}")
    print(f"elapsed_sec: {elapsed:.2f}")


if __name__ == "__main__":
    run("fixed")
    run("semantic")