"""Hash-based similarity using perceptual average hash (aHash).

This metric computes a compact 64-bit (for hash_size=8) perceptual hash.
We search by scanning the database and computing the normalized Hamming
similarity. ANN is typically unnecessary for 64-bit hashes.

DB expectation:
- Table `images` has a column `image_hash` that stores a pickled
  `imagehash.ImageHash` object (or you can switch to hex strings if preferred).
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
from PIL import Image
import imagehash


class HashingSimilarity:
    """Perceptual hashing (average hash) with DB-scan search."""

    def __init__(self, loader: Any, hash_size: int = 8) -> None:
        """
        Parameters
        ----------
        loader : Any
            ImageLoader instance providing `loader.db` for DB access.
        hash_size : int
            Hash grid size (8 → 64-bit hash).
        """
        self.loader = loader
        self.hash_size = hash_size
        self._bits = hash_size * hash_size  # total number of bits

    # --------------------------- Feature extraction ---------------------------

    @staticmethod
    def _ensure_pil(image: Any) -> Image.Image:
        """Convert supported inputs (path/np.ndarray/PIL) to a PIL.Image."""
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            return Image.open(image)
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        raise TypeError("Supported input types: PIL.Image, str (path), numpy.ndarray")

    def compute_feature(self, image: Any) -> imagehash.ImageHash:
        """
        Compute a perceptual average hash for the given image.
        """
        img = self._ensure_pil(image).convert("RGB")
        return imagehash.average_hash(img, hash_size=self.hash_size)

    # --------------------------------- Search --------------------------------

    def _similarity(self, h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> float:
        """Normalized Hamming similarity ∈ [0, 1]."""
        dist = (h1 - h2)  # Hamming distance (0..bits)
        return 1.0 - (dist / float(self._bits))

    def find_similar(self, query_hash: imagehash.ImageHash, best_k: int = 5) -> list[int]:
        """
        Return top-k image IDs most similar by normalized Hamming similarity.
        """
        similarities: list[tuple[int, float]] = []

        cur = self.loader.db.cursor
        cur.execute("SELECT image_id, image_hash FROM images WHERE image_hash IS NOT NULL;")
        rows = cur.fetchall()

        for idx, (image_id, hash_blob) in enumerate(rows):
            db_hash = pickle.loads(hash_blob)  # stored as pickled ImageHash
            sim = self._similarity(query_hash, db_hash)
            similarities.append((image_id, sim))

            if idx % 100 == 0:
                print(f"Compared: {idx} images")

        similarities.sort(key=lambda t: t[1], reverse=True)
        return [img_id for img_id, _ in similarities[:best_k]]