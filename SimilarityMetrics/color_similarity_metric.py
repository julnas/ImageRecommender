"""
Color-based similarity with FAISS-HNSW acceleration.

Dieses Modul berechnet RGB- und HSL-Histogramme pro Bild, normalisiert diese
und fasst sie zu einem einzelnen Feature-Vektor zusammen. Für schnelles Suchen
wird ein FAISS-HNSW-Index verwendet. Falls kein Index geladen wurde, erfolgt
ein Fallback zu einer direkten Ähnlichkeitsberechnung über die Datenbank.

Datenbank-Anforderung:
- Tabelle `images` muss eine Spalte `color_histogram` enthalten,
  in der ein Pickle-Objekt gespeichert ist:
  (rgb_hist, hsl_hist), wobei beide Tupel aus drei numpy-Arrays bestehen.
"""

import pickle
from typing import Any, Tuple
import faiss
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine


class ColorSimilarity:
    """Farbähnlichkeit basierend auf RGB- und HSL-Histogrammen mit FAISS-HNSW."""

    def __init__(self, loader: Any, bins: int = 16) -> None:
        """
        Parameters
        ----------
        loader : Any
            Instanz von ImageLoader (muss Zugriff auf loader.db haben).
        bins : int
            Anzahl der Bins pro Kanal (RGB und H, S, L).
        """
        self.loader = loader
        self.bins = bins
        self._dim = bins * 6  # r, g, b, h, s, l → insgesamt 6 Vektorsegmente
        self.faiss_index = None  # FAISS-Index-Objekt

    # --------------------------- Feature-Extraktion ---------------------------

    def _ensure_pil(self, image: Any) -> Image.Image:
        """Konvertiert Pfad / ndarray / PIL in ein PIL.Image."""
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            return Image.open(image)
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        raise TypeError("Unterstützte Typen: PIL.Image, str (Pfad), numpy.ndarray")

    def calculate_rgb_histogram(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Berechnet normalisierte RGB-Histogramme auf einem BGR-Bild (OpenCV)."""
        hist_r = cv2.calcHist([image_bgr], [2], None, [self.bins], [0, 256])
        hist_g = cv2.calcHist([image_bgr], [1], None, [self.bins], [0, 256])
        hist_b = cv2.calcHist([image_bgr], [0], None, [self.bins], [0, 256])
        for h in (hist_r, hist_g, hist_b):
            s = float(h.sum())
            h /= s if s > 0 else 1.0
        return hist_r, hist_g, hist_b

    def calculate_hsl_histogram(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Berechnet normalisierte H, S, L-Histogramme (OpenCV verwendet HLS-Reihenfolge)."""
        hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
        hist_h = cv2.calcHist([hls], [0], None, [self.bins], [0, 180])  # Hue 0..179
        hist_l = cv2.calcHist([hls], [1], None, [self.bins], [0, 256])
        hist_s = cv2.calcHist([hls], [2], None, [self.bins], [0, 256])
        for h in (hist_h, hist_s, hist_l):
            s = float(h.sum())
            h /= s if s > 0 else 1.0
        return hist_h, hist_s, hist_l

    def compute_feature(self, image: Any) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """Berechnet das Farbfeature (RGB- und HSL-Histogramme) für ein Bild."""
        img = self._ensure_pil(image).convert("RGB")
        img_np = np.asarray(img, dtype=np.uint8)
        image_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        rgb_hist = self.calculate_rgb_histogram(image_bgr)
        hsl_hist = self.calculate_hsl_histogram(image_bgr)
        return rgb_hist, hsl_hist

    def _feature_to_vec(self, feature: Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]) -> np.ndarray:
        """Wandelt (rgb_hist, hsl_hist) in einen normalisierten 1D-Feature-Vektor um."""
        (r, g, b), (h, s, l) = feature
        vec = np.hstack([
            r.flatten(), g.flatten(), b.flatten(),
            h.flatten(), s.flatten(), l.flatten()
        ]).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Berechnet die Kosinus-Ähnlichkeit zwischen zwei Histogrammen."""
        return 1.0 - float(cosine(a.ravel(), b.ravel()))

    # --------------------------- FAISS-HNSW:Load --------------------------

    def load_faiss_hnsw_index(self, index_path: str, ef: int = 100) -> None:
        """Lädt einen gespeicherten FAISS-HNSW-Index und setzt efSearch."""
        idx = faiss.read_index(index_path)
        # efSearch für schnelles Suchen setzen
        if hasattr(idx, "hnsw"):
            idx.hnsw.efSearch = ef
        elif hasattr(idx, "index") and hasattr(idx.index, "hnsw"):  # IDMap2 → innerer Index
            idx.index.hnsw.efSearch = ef
        self.faiss_index = idx
        print(f"[FAISS-HNSW] Index geladen von {index_path} (efSearch={ef})")

    # -------------------------------- Suche --------------------------------

    def find_similar(self, query_feature: Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]], best_k: int = 5) -> list[int]:
        """
        Findet die ähnlichsten Bilder basierend auf dem geladenen FAISS-HNSW-Index.
        Falls kein Index geladen ist, erfolgt ein Fallback zu einer DB-Suche.
        """
        if self.faiss_index is not None:
            q = self._feature_to_vec(query_feature).reshape(1, -1)
            scores, ids = self.faiss_index.search(q, best_k)
            return [int(i) for i in ids[0] if i != -1]

        # Fallback: langsame DB-Suche mit Cosinus-Ähnlichkeit
        cur = self.loader.db.cursor
        cur.execute(
            "SELECT image_id, color_histogram FROM images WHERE color_histogram IS NOT NULL;"
        )
        rows = cur.fetchall()
        sims = []
        q_rgb, q_hsl = query_feature
        for image_id, blob in rows:
            rgb_hsl = pickle.loads(blob)
            rgb_sim = sum(self._cosine_sim(qc, ic) for qc, ic in zip(q_rgb, rgb_hsl[0])) / 3
            hsl_sim = sum(self._cosine_sim(qc, ic) for qc, ic in zip(q_hsl, rgb_hsl[1])) / 3
            sims.append((image_id, (rgb_sim + hsl_sim) / 2))

        sims.sort(key=lambda x: x[1], reverse=True)
        return [img_id for img_id, _ in sims[:best_k]]
