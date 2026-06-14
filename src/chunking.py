from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 64,
) -> list[str]:
    """按字符数切分中文长文档（简单稳健，适合简历项目）。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start += step
    return chunks
