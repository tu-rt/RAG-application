from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.chunking import chunk_text
from src.embedding import EmbeddingModel


@dataclass
class RetrievedChunk:
    text: str
    score: float
    chunk_id: int


class InMemoryRetriever:
    """单篇文档内的向量检索（MultiFieldQA 闭卷设定）。"""

    def __init__(self, embedder: EmbeddingModel) -> None:
        self.embedder = embedder
        self._chunks: list[str] = []
        self._vectors: Optional[np.ndarray] = None

    def build_index(self, document: str, chunk_size: int, overlap: int) -> int:
        self._chunks = chunk_text(document, chunk_size=chunk_size, overlap=overlap)
        if not self._chunks:
            self._vectors = None
            return 0
        self._vectors = self.embedder.encode(self._chunks)
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self._chunks or self._vectors is None:
            return []
        q = self.embedder.encode([query])[0]
        scores = self._vectors @ q
        k = min(top_k, len(self._chunks))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [
            RetrievedChunk(text=self._chunks[i], score=float(scores[i]), chunk_id=int(i))
            for i in idx
        ]


class Reranker:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        cache_folder: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        print(f"[reranker] 加载模型: {model_name}")
        kwargs: dict = {"device": device}
        if cache_folder:
            kwargs["cache_folder"] = cache_folder
        self.model = CrossEncoder(model_name, **kwargs)

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int = 3
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [[query, c.text] for c in chunks]
        scores = self.model.predict(pairs)
        order = np.argsort(-np.asarray(scores))[: min(top_k, len(chunks))]
        return [chunks[i] for i in order]
