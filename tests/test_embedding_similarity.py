import numpy as np
import pickle

import torch

from SimilarityMetrics.embeddings_similarity_metric import EmbeddingSimilarity


class DummyLoader:
    class DummyDB:
        def __init__(self):
            self.cursor = self

        def execute(self, query):
            pass

        def fetchall(self):
            # Use non-zero, orthogonal vectors so cosine similarity is defined.
            emb1 = np.zeros(512, dtype=np.float32)
            emb1[0] = 1.0
            emb2 = np.zeros(512, dtype=np.float32)
            emb2[1] = 1.0
            return [
                (1, pickle.dumps(emb1)),
                (2, pickle.dumps(emb2)),
            ]

    def __init__(self):
        self.db = self.DummyDB()


class DummyEmbeddingModel:
    def __call__(self, batch):
        vector = torch.arange(1, 513, dtype=torch.float32)
        return vector.reshape(1, 512, 1, 1).repeat(batch.shape[0], 1, 1, 1)


def test_compute_feature_shape_and_norm():
    metric = EmbeddingSimilarity(loader=DummyLoader(), normalize=True)
    metric.model = DummyEmbeddingModel()
    metric.transform = lambda image: torch.ones((3, 224, 224), dtype=torch.float32)

    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    feat = metric.compute_feature(img)

    assert feat.shape == (512,)
    norm = np.linalg.norm(feat)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_find_similar_db_scan():
    metric = EmbeddingSimilarity(loader=DummyLoader(), normalize=True)
    query_vec = np.zeros(512, dtype=np.float32)
    query_vec[0] = 1.0

    ids = metric.find_similar(query_vec, best_k=1)

    assert ids == [1]


def test_find_similar_returns_k_results():
    metric = EmbeddingSimilarity(loader=DummyLoader(), normalize=True)
    query_vec = np.zeros(512, dtype=np.float32)
    query_vec[0] = 1.0

    ids = metric.find_similar(query_vec, best_k=2)

    assert ids == [1, 2]
