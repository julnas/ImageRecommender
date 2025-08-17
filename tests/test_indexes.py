import numpy as np
import pytest

faiss = pytest.importorskip("faiss")


def test_ivfpq_small():
    X = np.random.rand(100, 32).astype("float32")
    Q = np.random.rand(5, 32).astype("float32")
    index = faiss.index_factory(32, "IVF16,PQ4", faiss.METRIC_L2)
    index.train(X)
    index.add(X)
    index.nprobe = 4
    D, I = index.search(Q, 3)
    assert I.shape == (5, 3)


def test_hnsw_small():
    X = np.random.rand(100, 32).astype("float32")
    Q = np.random.rand(5, 32).astype("float32")
    index = faiss.IndexHNSWFlat(32, 16)
    index.add(X)
    D, I = index.search(Q, 3)
    assert I.shape == (5, 3)
