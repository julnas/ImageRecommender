import os
import sqlite3

import numpy as np
import pytest
from PIL import Image

import setup_db
import build_indexes
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


def test_complete_setup_installs_database_and_indexes_together(
    tmp_path,
    monkeypatch,
):
    patch_feature_extractors(monkeypatch)

    drive_path = tmp_path / "drive"
    drive_path.mkdir()
    Image.new("RGB", (8, 8), "red").save(drive_path / "new.jpg")

    db_path = tmp_path / "images_database.db"
    create_existing_database(db_path)

    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    (index_dir / "state.txt").write_text("old")

    production_db_path = db_path
    production_index_dir = index_dir

    def fake_build_faiss_indexes(db_path, index_dir):
        assert os.path.abspath(db_path) != os.path.abspath(
            production_db_path
        )
        assert os.path.abspath(index_dir) != os.path.abspath(
            production_index_dir
        )

        connection = sqlite3.connect(db_path)
        rows = connection.execute(
            "SELECT file_path FROM images"
        ).fetchall()
        connection.close()
        assert rows == [("new.jpg",)]

        os.makedirs(index_dir)
        with open(os.path.join(index_dir, "state.txt"), "w") as file:
            file.write("new")
        return True

    monkeypatch.setattr(
        build_indexes,
        "build_faiss_indexes",
        fake_build_faiss_indexes,
    )

    succeeded = setup_db.run_offline_setup(
        base_dir=str(drive_path),
        db_path=str(db_path),
        index_dir=str(index_dir),
        commit_batch_size=1,
    )

    assert succeeded is True

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT file_path FROM images"
    ).fetchall()
    connection.close()

    assert rows == [("new.jpg",)]
    assert (index_dir / "state.txt").read_text() == "new"


def test_index_failure_preserves_database_and_indexes(
    tmp_path,
    monkeypatch,
):
    patch_feature_extractors(monkeypatch)

    drive_path = tmp_path / "drive"
    drive_path.mkdir()
    Image.new("RGB", (8, 8), "red").save(drive_path / "new.jpg")

    db_path = tmp_path / "images_database.db"
    create_existing_database(db_path)

    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    (index_dir / "state.txt").write_text("old")

    monkeypatch.setattr(
        build_indexes,
        "build_faiss_indexes",
        lambda db_path, index_dir: False,
    )

    succeeded = setup_db.run_offline_setup(
        base_dir=str(drive_path),
        db_path=str(db_path),
        index_dir=str(index_dir),
        commit_batch_size=1,
    )

    assert succeeded is False

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT file_path FROM images"
    ).fetchall()
    connection.close()

    assert rows == [("old_drive/old.jpg",)]
    assert (index_dir / "state.txt").read_text() == "old"


def test_limited_setup_does_not_build_or_replace_indexes(
    tmp_path,
    monkeypatch,
):
    patch_feature_extractors(monkeypatch)

    drive_path = tmp_path / "drive"
    drive_path.mkdir()
    Image.new("RGB", (8, 8), "red").save(drive_path / "new.jpg")

    db_path = tmp_path / "images_database.db"
    create_existing_database(db_path)

    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    (index_dir / "state.txt").write_text("old")

    def unexpected_index_build(*args, **kwargs):
        raise AssertionError(
            "Index building must not run for a limited scan."
        )

    monkeypatch.setattr(
        build_indexes,
        "build_faiss_indexes",
        unexpected_index_build,
    )

    succeeded = setup_db.run_offline_setup(
        base_dir=str(drive_path),
        db_path=str(db_path),
        index_dir=str(index_dir),
        max_images=1,
        commit_batch_size=1,
    )

    assert succeeded is False

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT file_path FROM images"
    ).fetchall()
    connection.close()

    assert rows == [("old_drive/old.jpg",)]
    assert (index_dir / "state.txt").read_text() == "old"


def test_install_failure_restores_both_previous_artifacts(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "images_database.db"
    db_path.write_text("old database")

    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    (index_dir / "state.txt").write_text("old indexes")

    staged_db_path = tmp_path / "staged.db"
    staged_db_path.write_text("new database")

    staged_index_dir = tmp_path / "staged_indexes"
    staged_index_dir.mkdir()
    (staged_index_dir / "state.txt").write_text("new indexes")

    real_replace = os.replace
    failed = False

    def fail_when_installing_indexes(source, destination):
        nonlocal failed

        if (
            not failed
            and os.path.abspath(source)
            == os.path.abspath(staged_index_dir)
        ):
            failed = True
            raise OSError("simulated index installation failure")

        return real_replace(source, destination)

    monkeypatch.setattr(
        setup_db.os,
        "replace",
        fail_when_installing_indexes,
    )

    with pytest.raises(OSError):
        setup_db._install_database_and_indexes(
            staged_db_path=str(staged_db_path),
            staged_index_dir=str(staged_index_dir),
            db_path=str(db_path),
            index_dir=str(index_dir),
        )

    assert db_path.read_text() == "old database"
    assert (index_dir / "state.txt").read_text() == "old indexes"
