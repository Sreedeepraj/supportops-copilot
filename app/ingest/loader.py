from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_markdown_files(root_dir: str) -> list[dict]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root_dir}")

    documents = []
    for file in root.rglob("*.md"):
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                documents.append({"text": text, "source": str(file)})
        except Exception as e:
            logger.warning(f"Failed to load {file}: {e}")

    logger.info(f"Loaded {len(documents)} markdown documents from {root_dir}")
    return documents