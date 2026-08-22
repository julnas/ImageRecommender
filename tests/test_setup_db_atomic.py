import sqlite3

import numpy as np
from PIL import Image

import setup_db
from image_recommender.database import Database


class DummyColorSimilarity:
    def __init__(self, loader):
        self.loader = loader

    def compute_feature(self, image):
        channel = np.array([1.0], dtype=np.float32)
        return ((channel, channel, channel), (channel, channel, channel))


class DummyEmbeddingSimilarity:
    def __init__(self, loader):
        self.loader = loader

    def compute_feature(self, image):
        return np.array([1.0, 0.0], dtype=np.float32)


class DummyHashingSimilarity:
    def __init__(self, loader):
        self.loader = loader

    def compute_feature(self, image):
        return object()

    def hash_to_blob(self, hash_value):
        return b"\x00" * 8


def patch_feature_extractors(monkeypatch):
    monkeypatch.setattr(setup_db, "ColorSimilarity", DummyColorSimilarity)
    monkeypatch.setattr(setup_db, "EmbeddingSimilarity", DummyEmbeddingSimilarity)
    monkeypatch.setattr(setup_db, "HashingSimilarity", DummyHashingSimilarity)


def create_existing_database(db_path):
    db = Database(str(db_path))
    setup_db.create_database_schema(db)
    db.cursor.execute(
        """
        INSERT INTO images (file_path, image_hash)
        VALUES (?, ?)
        """,
        ("old_drive/old.jpg", b"\x01" * 8),
    )
    db.connection.commit()
    db.close()


def test_full_scan_replaces_old_drive_entries(tmp_path, monkeypatch):
    patch_feature_extractors(monkeypatch)
    drive_path = tmp_path / "drive"
    drive_path.mkdir()
    Image.new("RGB", (8, 8), "red").save(drive_path / "new.jpg")

    db_path = tmp_path / "images_database.db"
    create_existing_database(db_path)

    succeeded = setup_db.scan_and_fill_database(
        base_dir=str(drive_path),
        db_path=str(db_path),
        commit_batch_size=1,
    )

    assert succeeded is True
    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT file_path, length(image_hash) FROM images ORDER BY image_id"
    ).fetchall()
    connection.close()
    assert rows == [("new.jpg", 8)]


def test_scan_failure_preserves_existing_database(tmp_path, monkeypatch):
    patch_feature_extractors(monkeypatch)
    drive_path = tmp_path / "drive"
    drive_path.mkdir()
    db_path = tmp_path / "images_database.db"
    create_existing_database(db_path)

    def failing_scan(*args, **kwargs):
        raise RuntimeError("simulated drive failure")
        yield

    monkeypatch.setattr(setup_db, "iter_image_paths", failing_scan)

    succeeded = setup_db.scan_and_fill_database(
        base_dir=str(drive_path),
        db_path=str(db_path),
    )

    assert succeeded is False
    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT file_path, image_hash FROM images"
    ).fetchall()
    connection.close()
    assert rows == [("old_drive/old.jpg", b"\x01" * 8)]
