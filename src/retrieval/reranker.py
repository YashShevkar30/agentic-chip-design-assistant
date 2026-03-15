"""Cross-encoder reranking for document relevance refinement."""
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Reranks retrieved documents using cross-encoder scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder loaded: %s", self.model_name)
        except ImportError:
            logger.warning("sentence-transformers not available, using heuristic reranking")

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []

        if self._model is not None:
            return self._model_rerank(query, documents, top_k)
        return self._heuristic_rerank(query, documents, top_k)

    def _model_rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        pairs = [(query, doc.get("text", str(doc))) for doc in documents]
        scores = self._model.predict(pairs)
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [
            {**doc, "rerank_score": round(float(score), 4)}
            for score, doc in scored_docs[:top_k]
        ]

    def _heuristic_rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        query_terms = set(query.lower().split())
        scored = []
        for doc in documents:
            text = doc.get("text", str(doc)).lower()
            overlap = sum(1 for term in query_terms if term in text)
            length_bonus = min(len(text) / 1000, 0.5)
            score = overlap / (len(query_terms) + 1e-8) + length_bonus
            scored.append((score, doc))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [{**doc, "rerank_score": round(float(s), 4)} for s, doc in scored[:top_k]]
