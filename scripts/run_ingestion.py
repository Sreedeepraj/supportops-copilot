from dotenv import load_dotenv
load_dotenv()


from app.ingest.pipeline import run_ingestion

if __name__ == "__main__":
    records = run_ingestion("data/docs/langchain/langchain", max_docs=10)
    print(f"records={len(records)}")