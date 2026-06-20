from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


class LocalEmbeddingModel:
    def __init__(self, model_dir: Path, *, model_name: str) -> None:
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Local embedding model directory not found: {self.model_dir}. "
                "Populate this folder before running indexing or search."
            )
        self._model = SentenceTransformer(str(self.model_dir), local_files_only=True)

    def encode_passages(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        return self._encode([f"passage: {text}" for text in texts], batch_size=batch_size)

    def encode_queries(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        return self._encode([f"query: {text}" for text in texts], batch_size=batch_size)

    def _encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class LocalRerankerModel:
    def __init__(self, model_dir: Path, *, model_name: str) -> None:
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Local reranker model directory not found: {self.model_dir}. "
                "Populate this folder before using cross-encoder reranking."
            )
        self._model = CrossEncoder(str(self.model_dir), local_files_only=True)

    def predict(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.empty((0,), dtype=np.float32)
        pairs = [(query, document) for document in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32).reshape(-1)


@lru_cache(maxsize=2)
def load_embedding_model(model_dir: str, model_name: str) -> LocalEmbeddingModel:
    return LocalEmbeddingModel(Path(model_dir), model_name=model_name)


@lru_cache(maxsize=1)
def load_reranker_model(model_dir: str, model_name: str) -> LocalRerankerModel:
    return LocalRerankerModel(Path(model_dir), model_name=model_name)
