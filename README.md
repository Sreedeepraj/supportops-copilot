# SupportOps Copilot

A small FastAPI service that demonstrates a docs RAG pipeline and a simple agentic retrieval flow. It ingests markdown docs into a local Chroma vector store, retrieves relevant chunks, and answers questions with citations.

**Status:** early prototype

**What’s here**
- FastAPI app with structured logging, request IDs, and error handling.
- RAG endpoint: retrieve + answer with citations and latency stats.
- Agent endpoint: retrieve → assess → rewrite → retrieve → answer.
- Ingestion pipeline for markdown docs with fixed and semantic chunking.
- Retry policy for transient upstream failures.

## Architecture

**RAG flow**
1. Load vector store (Chroma persisted in `data/vector_store`).
2. Retrieve top-k chunks for the question.
3. Build context and answer via OpenAI chat model.
4. Return answer + citations + timing stats.

**Agent flow**
1. Retrieve.
2. Assess quality via a keyword heuristic.
3. If weak, rewrite the query using the LLM.
4. Retrieve again and answer.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# create .env with OPENAI_API_KEY
cp .env .env.local  # optional
```

```bash
uvicorn main:app --reload
```

Open: `http://localhost:8000/v1/health`

## Environment

Required:
- `OPENAI_API_KEY`

Optional:
- `APP_ENV` (dev | stage | prod)
- `LOG_LEVEL`, `LOG_APP_LEVEL`, `LOG_RAG_LEVEL`
- `HTTP_TIMEOUT_SECONDS`, `RETRY_MAX_ATTEMPTS`, `RETRY_BASE_DELAY_SECONDS`

## Ingestion

Docs are loaded from `data/docs/langchain/langchain` by default.

```bash
# build vector store from a sample of docs
python scripts/build_vector_store.py
```

Other utilities:
- `python scripts/run_ingestion.py`
- `python scripts/compare_chunking.py`
- `python scripts/test_retrieval.py`

## API

Base path: `/v1`

**Health**
- `GET /health`
- `GET /error` (returns a controlled AppError)
- `GET /crash` (forces an unhandled exception)
- `GET /retry-test` (demonstrates retry behavior)

**RAG**
- `POST /qa`

Request:
```json
{
  "question": "How do agents work in LangChain?",
  "top_k": 4
}
```

Response:
```json
{
  "answer": "...",
  "citations": [{"id": "...", "source": "...", "score": 0.12}],
  "stats": {
    "retrieved": 4,
    "latency_ms": {"vector_store_init": 10, "retrieval": 120, "llm": 900, "total": 1030},
    "tokens": {"prompt_tokens": 123, "completion_tokens": 45}
  }
}
```

**Agent**
- `POST /qa-agent`

Response includes:
- `path`: `direct` or `rewrite`
- `rewritten_question` when a rewrite happens
- `stats` with step-level latency and token usage

## Notes

- The vector store is created locally in `data/vector_store`.
- Chunking supports both fixed-size and semantic splitting.
- Retry logic only retries `RetryableError`.

## Roadmap ideas

- Cache vector store initialization.
- Add timeouts and retry caps for embedding calls.
- Expand ingestion to more docs and add metadata filtering.
