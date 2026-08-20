"""Hash-based similarity using perceptual average hash (aHash).

This metric computes a compact 64-bit perceptual hash for hash_size=8.

Hashes are stored in the database as compact 8-byte BLOBs instead of
pickled imagehash.ImageHash objects.

Similarity search is performed by:
- loading hashes into a NumPy uint64 array
- calculating Hamming distances using vectorized XOR operations
- using partial sorting to return only the best matches

DB expectation:
- Table `images` contains an `image_hash` column.
- `image_hash` stores the hash as an 8-byte unsigned integer BLOB.
"""

from __future__ import annotations

from typing import Any

import imagehash
import numpy as np
from PIL import Image


class HashingSimilarity:
    """Perceptual hashing (average hash) with vectorized DB-scan search."""

    def __init__(
        self,
        loader: Any,
        hash_size: int = 8,
    ) -> None:
        """
        Parameters
        ----------
        loader : Any
            ImageLoader instance providing `loader.db` for DB access.

        hash_size : int
            Hash grid size.

            hash_size=8 produces a 64-bit hash and therefore
            an 8-byte database representation.
        """

        self.loader = loader
        self.hash_size = hash_size
        self._bits = hash_size * hash_size
        self._cached_image_ids: np.ndarray | None = None
        self._cached_hashes: np.ndarray | None = None

        if self._bits > 64:
            raise ValueError(
                "hash_size must produce at most 64 bits."
            )

    # --------------------------- Feature extraction ---------------------------

    @staticmethod
    def _ensure_pil(image: Any) -> Image.Image:
        """Convert supported inputs to a PIL.Image."""

        if isinstance(image, Image.Image):
            return image

        if isinstance(image, str):
            return Image.open(image)

        if isinstance(image, np.ndarray):
            return Image.fromarray(image)

        raise TypeError(
            "Supported input types: "
            "PIL.Image, str (path), numpy.ndarray"
        )

    def compute_feature(
        self,
        image: Any,
    ) -> imagehash.ImageHash:
        """
        Compute a perceptual average hash for the given image.
        """

        img = self._ensure_pil(image).convert("RGB")

        return imagehash.average_hash(
            img,
            hash_size=self.hash_size,
        )

    # --------------------------- Hash conversion ---------------------------

    @staticmethod
    def hash_to_blob(
        hash_value: imagehash.ImageHash,
    ) -> bytes:
        """
        Convert an ImageHash into a compact 8-byte BLOB.

        The hash bits are represented as a uint64 value.
        """

        hash_array = np.asarray(
            hash_value.hash,
            dtype=np.uint8,
        )

        bits = hash_array.flatten()

        if len(bits) > 64:
            raise ValueError(
                "Hash contains more than 64 bits."
            )

        value = 0

        for bit in bits:
            value = (value << 1) | int(bit)

        return value.to_bytes(8, byteorder="big", signed=False)

    @staticmethod
    def blob_to_uint64(
        hash_blob: bytes,
    ) -> np.uint64:
        """
        Convert an 8-byte database BLOB into a NumPy uint64.
        """

        if len(hash_blob) != 8:
            raise ValueError(
                f"Invalid hash size: expected 8 bytes, "
                f"got {len(hash_blob)} bytes."
            )

        return np.uint64(
            int.from_bytes(hash_blob, byteorder="big", signed=False)
        )

    @staticmethod
    def hash_to_uint64(
        hash_value: imagehash.ImageHash,
    ) -> np.uint64:
        """
        Convert an ImageHash directly into a uint64 value.
        """

        blob = HashingSimilarity.hash_to_blob(hash_value)
        return HashingSimilarity.blob_to_uint64(blob)

    # --------------------------- Similarity ---------------------------

    def _similarity(
        self,
        h1: imagehash.ImageHash,
        h2: imagehash.ImageHash,
    ) -> float:
        """Return normalized Hamming similarity in [0, 1]."""

        dist = h1 - h2

        return 1.0 - (
            dist / float(self._bits)
        )

    # --------------------------- Search ---------------------------

    def clear_cache(self) -> None:
        """Discard cached database hashes after external database changes."""

        self._cached_image_ids = None
        self._cached_hashes = None

    def load_cache(self) -> int:
        """Load database hashes now and return the number of cached entries."""

        image_ids, _ = self._load_hash_cache()
        return int(image_ids.size)

    def _load_hash_cache(self) -> tuple[np.ndarray, np.ndarray]:
        """Load compact hashes from SQLite once and retain them in memory."""

        if self._cached_image_ids is not None and self._cached_hashes is not None:
            return self._cached_image_ids, self._cached_hashes

        cur = self.loader.db.cursor
        cur.execute(
            """
            SELECT image_id, image_hash
            FROM images
            WHERE image_hash IS NOT NULL
            ORDER BY image_id ASC
            """
        )
        rows = cur.fetchall()

        image_ids = np.fromiter(
            (image_id for image_id, _ in rows),
            dtype=np.int64,
            count=len(rows),
        )
        compact_blobs = []

        for image_id, hash_blob in rows:
            try:
                if len(hash_blob) != 8:
                    raise ValueError
                compact_blobs.append(bytes(hash_blob))
            except (TypeError, ValueError) as error:
                blob_length = len(hash_blob) if hash_blob is not None else 0
                raise ValueError(
                    f"Invalid hash BLOB for image_id {image_id}: expected "
                    f"the optimized 8-byte format, got {blob_length} bytes. "
                    "Rebuild the database with setup_db.py."
                ) from error

        if compact_blobs:
            hashes = np.frombuffer(
                b"".join(compact_blobs),
                dtype=">u8",
            ).astype(np.uint64)
        else:
            hashes = np.empty(0, dtype=np.uint64)

        self._cached_image_ids = image_ids
        self._cached_hashes = hashes
        return image_ids, hashes

    def find_similar(
        self,
        query_hash: imagehash.ImageHash,
        best_k: int = 5,
    ) -> list[int]:
        """
        Return the top-k image IDs by Hamming similarity.

        Hashes are loaded as uint64 values and compared using
        vectorized XOR operations.

        Only the best `best_k` results are selected using
        partial sorting instead of sorting the complete result set.
        """

        if best_k <= 0:
            return []

        query_value = self.hash_to_uint64(
            query_hash
        )

        image_ids, hashes = self._load_hash_cache()

        if image_ids.size == 0:
            return []

        # XOR identifies all differing bits.
        xor_values = np.bitwise_xor(
            hashes,
            query_value,
        )

        # Count set bits to get the Hamming distance.
        distances = np.bitwise_count(xor_values)

        # We only need the best `best_k` results.
        k = min(
            best_k,
            len(distances),
        )

        # Partial selection instead of sorting all results.
        best_indices = np.argpartition(
            distances,
            k - 1,
        )[:k]

        # Sort only the selected top-k results.
        best_indices = best_indices[
            np.argsort(
                distances[best_indices]
            )
        ]

        return [
            int(image_ids[index])
            for index in best_indices
        ]
