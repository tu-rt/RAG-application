from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        cache_folder: Optional[str] = None,
    ) -> None:
        print(f"[embedding] 加载模型: {model_name}")
        kwargs: dict = {"device": device}
        if cache_folder:
            kwargs["cache_folder"] = cache_folder
        self.model = SentenceTransformer(model_name, **kwargs)
        self.device = device

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)
