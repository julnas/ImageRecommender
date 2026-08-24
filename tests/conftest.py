import sqlite3

import pytest
from PIL import Image

from image_recommender.database import Database


@pytest.fixture
def image_store(tmp_path):
    """Provide a self-contained image directory and SQLite database."""

    image_dir = tmp_path / "images"
    image_dir.mkdir()

    test_images = [
        (1, "red.png", (255, 0, 0)),
        (2, "green.png", (0, 255, 0)),
        (3, "blue.png", (0, 0, 255)),
    ]

    for _, filename, color in test_images:
        Image.new("RGB", (16, 12), color).save(image_dir / filename)

    db_path = tmp_path / "images.db"
    connection = sqlite3.connect(db_path)
    connection.execute("""
        CREATE TABLE images (
            image_id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL UNIQUE,
            color_histogram BLOB,
            embedding BLOB,
            image_hash BLOB,
            width INTEGER,
            height INTEGER,
            file_size INTEGER
        )
        """)
    connection.executemany(
        """
        INSERT INTO images (image_id, file_path, width, height)
        VALUES (?, ?, ?, ?)
        """,
        [(image_id, filename, 16, 12) for image_id, filename, _ in test_images],
    )
    connection.commit()
    connection.close()

    db = Database(str(db_path))

    try:
        yield {
            "db": db,
            "db_path": db_path,
            "image_dir": image_dir,
            "image_ids": [1, 2, 3],
        }
    finally:
        db.close()
