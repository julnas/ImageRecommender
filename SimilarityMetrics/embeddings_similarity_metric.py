"""Embedding-based similarity with optional FAISS IVFPQ acceleration.

This metric extracts a 512-D feature from a ResNet18 backbone. Vectors are
L2-normalized so cosine similarity equals inner product (IP). If a FAISS
IVFPQ index is available it will be used; otherwise we fall back to a
database scan with cosine similarity.

DB expectation:
- Table `images` has a column `embedding` that stores a pickled numpy array
  (float32, shape (512,)).
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "TRUE"  # macOS-Workaround gegen libomp-Doppelladung
)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.spatial.distance import cosine
from torchvision import models, transforms


class EmbeddingSimilarity:
    """Deep embedding similarity powered by a ResNet18 backbone."""

    def __init__(
        self,
        loader: Any,
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self.loader = loader
        self.device = device or ("cuda" if False else "cpu")
        self.normalize = normalize
        self.model = None
        self.transform = None
        self.faiss_index = None
        self.nprobe = 16

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

    def _ensure_model(self):
        """Lazy load the ResNet18 model and set it to evaluation mode."""
        if self.model is None:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            # optional: Threads begrenzen
            try:
                torch.set_num_threads(1)
            except Exception:
                pass

            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.model = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)
            self.model.to(self.device).eval()

            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def compute_feature(self, image: Any) -> np.ndarray:
        """
        Compute a 512-D float32 embedding (L2-normalized if enabled).
        """
        self._ensure_model()
        img = self._ensure_pil(image).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(x)  # (1,512,1,1)
        vec = feat.squeeze().float().cpu().numpy().astype(np.float32, copy=False)

        if self.normalize:
            n = float(np.linalg.norm(vec))
            if n > 0.0:
                vec /= n
        return vec

    # ----------------------------- IVFPQ build/load ---------------------------

    def build_ivfpq_index(
        self, index_path: str, nlist: int = 4096, m: int = 16
    ) -> None:
        """
        Build a FAISS IVFPQ index from all embeddings in the DB and persist it.

        Parameters
        ----------
        index_path : str
            File path to save the FAISS index.
        nlist : int
            Number of coarse clusters (IVF lists); ~sqrt(N) is a common heuristic.
        m : int
            Number of PQ sub-vectors (must divide the embedding dimension).
        """
        import faiss  # lazy import

        cur = self.loader.db.cursor
        cur.execute(
            "SELECT image_id, embedding FROM images WHERE embedding IS NOT NULL;"
        )
        rows = cur.fetchall()

        ids, vecs = [], []
        for image_id, blob in rows:
            v = pickle.loads(blob).astype(np.float32, copy=False)
            if self.normalize:
                n = float(np.linalg.norm(v))
                if n > 0.0:
                    v = v / n
            ids.append(image_id)
            vecs.append(v)

        x = np.vstack(vecs).astype(np.float32, copy=False)
        ids = np.asarray(ids, dtype=np.int64)
        d = x.shape[1]

        quantizer = faiss.IndexFlatIP(d)  # IP == cosine for normalized vectors
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 8 bits/code (default)
        index.train(x)
        index.add_with_ids(x, ids)
        index.nprobe = self.nprobe
        faiss.write_index(index, index_path)
        self.faiss_index = index

    def load_ivfpq_index(self, index_path: str) -> None:
        """Load a persisted FAISS IVFPQ index from disk."""
        import faiss  # lazy import

        index = faiss.read_index(index_path)
        index.nprobe = self.nprobe
        self.faiss_index = index

    # --------------------------------- Search --------------------------------

    def find_similar(self, query_vec: np.ndarray, best_k: int = 5) -> list[int]:
        """
        Return top-k image IDs similar to the given embedding.

        If a FAISS index is loaded, use it. Otherwise, perform a DB scan
        with cosine similarity.
        """
        # Ensure L2-normalization before searching (safety).
        if self.normalize:
            n = float(np.linalg.norm(query_vec))
            if n > 0.0:
                query_vec = (query_vec / n).astype(np.float32, copy=False)

        # Fast path: FAISS IVFPQ available.
        if self.faiss_index is not None:
            q = query_vec.reshape(1, -1).astype(np.float32, copy=False)
            scores, ids = self.faiss_index.search(q, best_k)
            return [int(i) for i in ids[0] if int(i) != -1]

        # Fallback: DB scan with cosine similarity.
        similarities: list[tuple[int, float]] = []

        cur = self.loader.db.cursor
        cur.execute(
            "SELECT image_id, embedding FROM images WHERE embedding IS NOT NULL;"
        )
        rows = cur.fetchall()

        for idx, (image_id, emb_blob) in enumerate(rows):
            v = pickle.loads(emb_blob).astype(np.float32, copy=False)
            if self.normalize:
                m = float(np.linalg.norm(v))
                if m > 0.0:
                    v = v / m
            sim = 1.0 - float(cosine(query_vec, v))
            similarities.append((image_id, sim))

        similarities.sort(key=lambda t: t[1], reverse=True)
        return [img_id for img_id, _ in similarities[:best_k]]
