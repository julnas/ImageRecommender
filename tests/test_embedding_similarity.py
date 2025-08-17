import numpy as np
import pytest
from SimilarityMetrics.embeddings_similarity_metric import EmbeddingSimilarity
import pickle


class DummyLoader:
    class DummyDB:
        def __init__(self):
            self.cursor = self

        def execute(self, query):
            pass

        def fetchall(self):
            # Return two dummy embeddings
            emb1 = np.ones(512, dtype=np.float32)
            emb2 = np.zeros(512, dtype=np.float32)
            return [
                (1, pickle.dumps(emb1)),
                (2, pickle.dumps(emb2)),
            ]

    def __init__(self):
        self.db = self.DummyDB()


def test_compute_feature_shape_and_norm():
    metric = EmbeddingSimilarity(loader=DummyLoader(), normalize=True)
    # Use a random numpy array as image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    feat = metric.compute_feature(img)
    assert feat.shape == (512,)
    norm = np.linalg.norm(feat)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_find_similar_db_scan():
    metric = EmbeddingSimilarity(loader=DummyLoader(), normalize=True)
    # Query vector similar to emb1
    query_vec = np.ones(512, dtype=np.float32)
    ids = metric.find_similar(query_vec, best_k=1)
    assert ids == [1]


def test_find_similar_returns_k_results():
    metric = EmbeddingSimilarity(loader=DummyLoader(), normalize=True)
    query_vec = np.ones(512, dtype=np.float32)
    ids = metric.find_similar(query_vec, best_k=2)
    assert set(ids) == {1, 2}
