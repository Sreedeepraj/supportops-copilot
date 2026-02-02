import logging
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def get_embedder():
    model = "text-embedding-3-small"
    logger.info(f"Using embedding model: {model}")
    return OpenAIEmbeddings(model=model)