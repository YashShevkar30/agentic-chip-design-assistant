"""Hybrid search combining dense and sparse retrieval."""
import re
import numpy as np
from collections import Counter
from typing import Optional

class BM25Scorer:
    """BM25 sparse keyword scoring implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.doc_lens = []
        self.avg_dl = 0
        self.corpus_size = 0
        self.tokenized_docs = []

    def fit(self, documents: list[str]):
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.doc_lens = [len(doc) for doc in self.tokenized_docs]
        self.avg_dl = sum(self.doc_lens) / (len(self.doc_lens) + 1e-8)
        self.corpus_size = len(documents)
        df = Counter()
        for doc in self.tokenized_docs:
            df.update(set(doc))
        self.doc_freqs = dict(df)

    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = self._tokenize(query)
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        tf = Counter(doc_tokens)
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            freq = tf[token]
            df = self.doc_freqs.get(token, 0)
            idf = np.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator
        return score

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


def hybrid_search(query: str, dense_results: list[dict], documents: list[str],
                  alpha: float = 0.6, top_k: int = 5) -> list[dict]:
    """Combine dense (semantic) and sparse (BM25) retrieval scores."""
    bm25 = BM25Scorer()
    bm25.fit(documents)

    combined = {}
    for result in dense_results:
        doc_text = result["document"].get("text", "")
        combined[doc_text] = {"dense_score": result["score"] * alpha, "result": result}

    for i, doc in enumerate(documents):
        sparse_score = bm25.score(query, i) * (1 - alpha)
        if doc in combined:
            combined[doc]["sparse_score"] = sparse_score
            combined[doc]["total"] = combined[doc]["dense_score"] + sparse_score
        else:
            combined[doc] = {"sparse_score": sparse_score, "total": sparse_score, "result": {"document": {"text": doc}, "score": sparse_score}}

    ranked = sorted(combined.values(), key=lambda x: x.get("total", 0), reverse=True)
    return [r["result"] for r in ranked[:top_k]]
