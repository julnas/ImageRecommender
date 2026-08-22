import imagehash
import numpy as np
import pytest
from PIL import Image

from SimilarityMetrics.hashing_similarity_metric import HashingSimilarity


class DummyCursor:
    def __init__(self, rows):
        self.rows = rows
        self.execute_count = 0

    def execute(self, query):
        self.execute_count += 1

    def fetchall(self):
        return self.rows


class DummyLoader:
    def __init__(self, rows):
        self.db = type("DummyDB", (), {"cursor": DummyCursor(rows)})()


def make_hash(enabled_bits=()):
    bits = np.zeros(64, dtype=bool)
    bits[list(enabled_bits)] = True
    return imagehash.ImageHash(bits.reshape(8, 8))


def test_hash_similarity_identity():
    img = Image.new("RGB", (16,16), (0,0,0))
    metric = HashingSimilarity(loader=None)
    h = metric.compute_feature(img)
    s = metric._similarity(h, h)
    assert abs(s - 1.0) < 1e-6


def test_hash_blob_is_exactly_eight_bytes_and_roundtrips():
    hash_value = make_hash((0, 5, 31, 63))

    blob = HashingSimilarity.hash_to_blob(hash_value)

    assert len(blob) == 8
    assert HashingSimilarity.blob_to_uint64(blob) == HashingSimilarity.hash_to_uint64(hash_value)


def test_hash_search_uses_compact_values_and_caches_database_rows():
    query_hash = make_hash()
    rows = [
        (30, HashingSimilarity.hash_to_blob(make_hash(range(64)))),
        (20, HashingSimilarity.hash_to_blob(make_hash((0,)))),
        (10, HashingSimilarity.hash_to_blob(query_hash)),
    ]
    loader = DummyLoader(rows)
    metric = HashingSimilarity(loader)

    assert metric.load_cache() == 3
    assert metric.find_similar(query_hash, best_k=2) == [10, 20]
    assert metric.find_similar(query_hash, best_k=1) == [10]
    assert loader.db.cursor.execute_count == 1

    metric.clear_cache()
    assert metric.find_similar(query_hash, best_k=1) == [10]
    assert loader.db.cursor.execute_count == 2


def test_hash_search_rejects_legacy_pickle_blob():
    metric = HashingSimilarity(DummyLoader([(1, b"not-eight-bytes")]))

    with pytest.raises(ValueError, match="Rebuild the database with setup_db.py"):
        metric.find_similar(make_hash(), best_k=1)
