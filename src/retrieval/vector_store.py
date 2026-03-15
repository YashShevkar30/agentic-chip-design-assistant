"""FAISS-based vector store for chip design documents."""
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FAISSStore:
    """Vector store using FAISS for similarity search over engineering documents."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._index = None
        self._documents = []
        self._ids = []
        self._build_index()

    def _build_index(self):
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            logger.info("FAISS index initialized (dim=%d)", self.dimension)
        except ImportError:
            logger.warning("FAISS not available, using numpy fallback")
            self._vectors = []

    def add_documents(self, docs: list[dict], embeddings: np.ndarray):
        if self._index is not None:
            import faiss
            faiss.normalize_L2(embeddings)
            self._index.add(embeddings)
        else:
            self._vectors.extend(embeddings.tolist())
        self._documents.extend(docs)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if self._index is not None:
            import faiss
            query = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query)
            scores, indices = self._index.search(query, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self._documents):
                    results.append({
                        "document": self._documents[idx],
                        "score": round(float(score), 4),
                    })
            return results
        return self._numpy_search(query_embedding, top_k)

    def _numpy_search(self, query: np.ndarray, top_k: int) -> list[dict]:
        if not self._vectors:
            return []
        vectors = np.array(self._vectors)
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        vec_norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        scores = vec_norms @ query_norm
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{"document": self._documents[i], "score": round(float(scores[i]), 4)} for i in top_indices]

    @property
    def size(self) -> int:
        return len(self._documents)
