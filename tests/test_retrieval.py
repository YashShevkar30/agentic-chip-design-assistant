"""Tests for retrieval components."""
import numpy as np
from src.retrieval.vector_store import FAISSStore

def test_vector_store_init():
    store = FAISSStore(dimension=64)
    assert store.size == 0

def test_add_and_search():
    store = FAISSStore(dimension=64)
    docs = [{"text": "chip design"}, {"text": "HDL optimization"}]
    embeddings = np.random.randn(2, 64).astype(np.float32)
    store.add_documents(docs, embeddings)
    assert store.size == 2
    query = np.random.randn(64).astype(np.float32)
    results = store.search(query, top_k=2)
    assert len(results) == 2
    assert "score" in results[0]
