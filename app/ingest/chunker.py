from langchain.text_splitter import RecursiveCharacterTextSplitter


def fixed_chunk(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break

    return chunks

def semantic_chunk(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> list[str]:
    """
    Recursive split tries to respect paragraph/section boundaries.
    Still character-based, but much better than raw slicing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]