"""Color-based similarity with optional HNSW acceleration.

This metric computes RGB and HSL histograms per image, normalizes them,
and (optionally) concatenates them into a single feature vector for HNSW.
If an index is available it will be used; otherwise we fall back to a
weighted per-channel cosine similarity computed by scanning the database.

DB expectation:
- Table `images` has a column `color_histogram` that stores a pickled tuple:
  (rgb_hist, hsl_hist), where each is a 3-tuple of numpy arrays.
"""

import pickle
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine


class ColorSimilarity:
    """Color-based similarity using RGB and HSL histograms."""

    def __init__(self, loader: Any, bins: int = 16) -> None:
        """
        Parameters
        ----------
        loader : Any
            ImageLoader instance providing `loader.db` for DB access.
        bins : int
            Number of bins per channel (RGB and H, S, L).
        """
        self.loader = loader
        self.bins = bins

        # Optional HNSW index (set via load_hnsw_index or build_hnsw_index).
        self.hnsw = None  # type: ignore[assignment]
        self._dim = bins * 6  # r, g, b, h, s, l → total features

    # --------------------------- Feature extraction ---------------------------

    def _ensure_pil(self, image: Any) -> Image.Image:
        """Convert supported inputs (path/np.ndarray/PIL) to a PIL.Image."""
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            return Image.open(image)
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        raise TypeError("Supported input types: PIL.Image, str (path), numpy.ndarray")

    def calculate_rgb_histogram(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Compute normalized RGB histograms on a BGR OpenCV image."""
        # OpenCV uses BGR channel order; map to R/G/B explicitly.
        hist_r = cv2.calcHist([image_bgr], [2], None, [self.bins], [0, 256])
        hist_g = cv2.calcHist([image_bgr], [1], None, [self.bins], [0, 256])
        hist_b = cv2.calcHist([image_bgr], [0], None, [self.bins], [0, 256])

        for h in (hist_r, hist_g, hist_b):
            s = float(h.sum())
            h /= s if s > 0 else 1.0
        return hist_r, hist_g, hist_b

    def calculate_hsl_histogram(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Compute normalized H, S, L histograms (OpenCV uses HLS order)."""
        hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
        hist_h = cv2.calcHist([hls], [0], None, [self.bins], [0, 180])  # Hue: 0..179
        hist_l = cv2.calcHist([hls], [1], None, [self.bins], [0, 256])
        hist_s = cv2.calcHist([hls], [2], None, [self.bins], [0, 256])

        for h in (hist_h, hist_s, hist_l):
            s = float(h.sum())
            h /= s if s > 0 else 1.0
        # Return order (H, S, L)
        return hist_h, hist_s, hist_l

    def compute_feature(self, image: Any) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """
        Compute the color feature as a tuple of histograms.

        Returns
        -------
        (rgb_hist, hsl_hist) :
            Each is a tuple of three numpy arrays (one per channel).
        """
        img = self._ensure_pil(image).convert("RGB")
        img_np = np.asarray(img, dtype=np.uint8)
        image_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        rgb_hist = self.calculate_rgb_histogram(image_bgr)
        hsl_hist = self.calculate_hsl_histogram(image_bgr)
        return rgb_hist, hsl_hist

    def _feature_to_vec(self, feature: Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]) -> np.ndarray:
        """Flatten (r,g,b,h,s,l) histograms into a single L2-normalized vector."""
        (r, g, b), (h, s, l) = feature
        vec = np.hstack(
            [r.ravel(), g.ravel(), b.ravel(), h.ravel(), s.ravel(), l.ravel()]
        ).astype(np.float32, copy=False)
        n = float(np.linalg.norm(vec))
        if n > 0.0:
            vec /= n
        return vec

    # ---------------------------- Similarity helpers --------------------------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between flattened histograms."""
        return 1.0 - float(cosine(a.ravel(), b.ravel()))

    # --------------------------- HNSW build / load ----------------------------

    def build_hnsw_index(
        self,
        index_path: str,
        m: int = 16,
        ef_construction: int = 200,
        ef: int = 100,
    ) -> None:
        """
        Build an HNSW index from all color histograms in the DB and persist it.

        Parameters
        ----------
        index_path : str
            File path to save the HNSW index.
        m : int
            HNSW graph connectivity (higher = better recall, more memory).
        ef_construction : int
            HNSW construction parameter (quality/speed trade-off).
        ef : int
            Query-time search breadth (higher = better recall, slower).
        """
        import hnswlib  # lazy import

        cur = self.loader.db.cursor
        cur.execute(
            "SELECT image_id, color_histogram FROM images "
            "WHERE color_histogram IS NOT NULL;"
        )
        rows = cur.fetchall()

        ids, vecs = [], []
        for image_id, blob in rows:
            rgb_hsl = pickle.loads(blob)
            vecs.append(self._feature_to_vec(rgb_hsl))
            ids.append(image_id)

        x = np.vstack(vecs).astype(np.float32, copy=False)
        ids = np.asarray(ids, dtype=np.int64)

        index = hnswlib.Index(space="cosine", dim=self._dim)
        index.init_index(max_elements=x.shape[0], ef_construction=ef_construction, M=m)
        index.add_items(x, ids)
        index.set_ef(ef)
        index.save_index(index_path)

        self.hnsw = index

    def load_hnsw_index(self, index_path: str, ef: int = 100) -> None:
        """Load a persisted HNSW index from disk."""
        import hnswlib  # lazy import

        index = hnswlib.Index(space="cosine", dim=self._dim)
        index.load_index(index_path)
        index.set_ef(ef)
        self.hnsw = index

    # --------------------------------- Search --------------------------------

    def find_similar(
        self,
        query_feature: Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]],
        best_k: int = 5,
    ) -> list[int]:
        """
        Return top-k image IDs similar to the given color feature.

        If an HNSW index is loaded, use it. Otherwise, perform a weighted
        cosine similarity scan per channel via the database.
        """
        # Fast path: HNSW available → search in vector space.
        if self.hnsw is not None:
            q = self._feature_to_vec(query_feature).reshape(1, -1)
            labels, _ = self.hnsw.knn_query(q, k=best_k)
            return [int(i) for i in labels[0]]

        # Fallback: DB scan with per-channel weighted scheme.
        similarities: list[tuple[int, float]] = []
        query_rgb, query_hsl = query_feature

        cur = self.loader.db.cursor
        cur.execute(
            "SELECT image_id, color_histogram FROM images "
            "WHERE color_histogram IS NOT NULL;"
        )
        rows = cur.fetchall()

        for idx, (image_id, hist_blob) in enumerate(rows):
            db_rgb, db_hsl = pickle.loads(hist_blob)

            r_sim = self._cosine_sim(query_rgb[0], db_rgb[0])
            g_sim = self._cosine_sim(query_rgb[1], db_rgb[1])
            b_sim = self._cosine_sim(query_rgb[2], db_rgb[2])
            rgb_sim = 0.30 * r_sim + 0.59 * g_sim + 0.11 * b_sim

            h_sim = self._cosine_sim(query_hsl[0], db_hsl[0])
            s_sim = self._cosine_sim(query_hsl[1], db_hsl[1])
            l_sim = self._cosine_sim(query_hsl[2], db_hsl[2])
            hsl_sim = 0.40 * h_sim + 0.30 * s_sim + 0.30 * l_sim

            total = 0.50 * rgb_sim + 0.50 * hsl_sim
            similarities.append((image_id, total))

            if idx % 100 == 0:
                print(f"Compared: {idx} images")

        similarities.sort(key=lambda t: t[1], reverse=True)
        return [img_id for img_id, _ in similarities[:best_k]]