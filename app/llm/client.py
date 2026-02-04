import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

def get_chat_model():
    # Start cheap + solid. You can switch later.
    model_name = "gpt-4o-mini"
    logger.info(f"Using chat model: {model_name}")
    return ChatOpenAI(model=model_name, temperature=0)