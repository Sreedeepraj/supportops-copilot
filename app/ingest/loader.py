from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SKIP_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
}

SKIP_DIRS = {".git", ".github", ".venv", "node_modules", "__pycache__"}


def load_markdown_files(root_dir: str, max_files: Optional[int] = None) -> list[dict]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root_dir}")

    documents: list[dict] = []
    for path in root.rglob("*.md"):
        # Skip hidden/system directories
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        # Skip noisy files
        if path.name in SKIP_FILES:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                documents.append({"text": text, "source": str(path)})
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

        if max_files and len(documents) >= max_files:
            break

    logger.info(f"Loaded {len(documents)} markdown documents from {root_dir}")
    return documents