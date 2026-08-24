import pickle
import sqlite3

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")


def test_ivfpq_small():
    rng = np.random.default_rng(42)
    X = rng.random((1024, 32), dtype=np.float32)
    Q = rng.random((5, 32), dtype=np.float32)
    index = faiss.index_factory(32, "IVF16,PQ4", faiss.METRIC_L2)
    index.train(X)
    index.add(X)
    index.nprobe = 4
    D, I = index.search(Q, 3)

    assert index.is_trained
    assert index.ntotal == 1024
    assert D.shape == (5, 3)
    assert I.shape == (5, 3)


def test_hnsw_small():
    rng = np.random.default_rng(42)
    X = rng.random((100, 32), dtype=np.float32)
    Q = rng.random((5, 32), dtype=np.float32)
    index = faiss.IndexHNSWFlat(32, 16)
    index.add(X)
    D, I = index.search(Q, 3)

    assert index.ntotal == 100
    assert D.shape == (5, 3)
    assert I.shape == (5, 3)


def test_project_index_builder_creates_readable_indexes(tmp_path, monkeypatch):
    import build_indexes

    db_path = tmp_path / "images.db"
    index_dir = tmp_path / "indexes"
    connection = sqlite3.connect(db_path)
    connection.execute("""
        CREATE TABLE images (
            image_id INTEGER PRIMARY KEY,
            color_histogram BLOB,
            embedding BLOB
        )
        """)

    rng = np.random.default_rng(7)

    for image_id in range(1, 257):
        embedding = rng.random(16, dtype=np.float32)
        channels = tuple(rng.random(2, dtype=np.float32) for _ in range(6))
        color_feature = (channels[:3], channels[3:])
        connection.execute(
            """
            INSERT INTO images (image_id, color_histogram, embedding)
            VALUES (?, ?, ?)
            """,
            (
                image_id,
                pickle.dumps(color_feature),
                pickle.dumps(embedding),
            ),
        )

    connection.commit()
    connection.close()

    monkeypatch.setattr(build_indexes, "DB_PATH", str(db_path))
    monkeypatch.setattr(build_indexes, "INDEX_DIR", str(index_dir))

    assert build_indexes.build_faiss_indexes() is True

    embedding_index = faiss.read_index(str(index_dir / "emb_ivfpq.faiss"))
    color_index = faiss.read_index(str(index_dir / "color_hnsw.faiss"))

    assert embedding_index.is_trained
    assert embedding_index.ntotal == 256
    assert color_index.ntotal == 256
    assert (index_dir / "emb_ivfpq.ids.pkl").is_file()
    assert (index_dir / "color_hnsw.ids.pkl").is_file()
