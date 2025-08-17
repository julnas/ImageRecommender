import os
import sqlite3
import pickle
import numpy as np
import faiss
from typing import List, Tuple

# -----------------------------
# Pfade & globale Defaults
# -----------------------------
DB_PATH = "images_database.db"
INDEX_DIR = "indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

# ---- IVFPQ (Embeddings) Defaults für ~250k Bilder ----
IVFPQ_DEFAULTS = {
    "nlist": 4096,           # deutlich höher als sqrt(N); gute Balance für 250k
    "target_code_bytes": 32, # grob 16–32 B üblich; 32 B = sehr ordentlich
    "nprobe": 96,            # 64–128: Recall/Latency-Tuning
    "use_opq": True,         # OPQ deutlich besserer Recall bei gleichem Code
    "train_samples": 200_000 # Trainingssample (statt auf allen Vektoren)
}

# ---- HNSW (Color-Histogramme) Defaults ----
HNSW_DEFAULTS = {
    "M": 32,                 # 16–32 üblich; 32 = höherer Recall
    "efConstruction": 300,   # 200–400 realistisch, statt 4096
    "efSearch": 256          # Query-Zeit-Recall; hier als Default gesetzt
}

RNG_SEED = 12345  # Reproduzierbarkeit für Training


# -----------------------------
# Hilfsfunktionen
# -----------------------------
def _choose_m(d: int, target_code_bytes: int) -> int:
    """
    Wähle m (PQ-Segmente) so, dass:
      - m * 1 byte ≈ target_code_bytes (da 8 bits pro Subquantizer)
      - d % m == 0
    Fallback: größter Teiler aus [64, 32, 16, 8, 4, 2] der d teilt und <= gewünschtem m.
    """
    desired_m = max(1, min(d, target_code_bytes))  # 8 bits → 1 Byte pro Subquantizer
    # runde desired_m auf "vernünftige" Kandidaten herunter (Powers of two sind praktisch)
    candidates = sorted({desired_m, 64, 32, 16, 8, 4, 2, 1}, reverse=True)
    # füge evtl. noch Divisoren von d um desired_m herum hinzu
    for m in range(min(desired_m, d), 0, -1):
        if d % m == 0:
            candidates.append(m)
            if len(candidates) > 50:
                break
    # dedupliziere & sortiere
    candidates = sorted(set(candidates), reverse=True)
    for m in candidates:
        if m <= d and d % m == 0:
            return m
    return 1  # Fallback, sollte nie erreicht werden


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    """Normalisiere Zeilen von X für Cosine/IP-Suche."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    np.divide(X, np.maximum(norms, 1e-12), out=X)
    return X


def _fetch_embeddings() -> Tuple[np.ndarray, np.ndarray]:
    """Hole (ids, embeddings) aus DB, normalisiert für Cosine/IP."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT image_id, embedding
        FROM images
        WHERE embedding IS NOT NULL
        ORDER BY image_id ASC
    """)
    rows = cur.fetchall()
    conn.close()

    ids: List[int] = []
    vecs: List[np.ndarray] = []
    for image_id, blob in rows:
        v = pickle.loads(blob).astype(np.float32, copy=False)
        ids.append(int(image_id))
        vecs.append(v)

    if not vecs:
        raise RuntimeError("No embeddings found in DB.")

    X = np.vstack(vecs).astype(np.float32, copy=False)
    X = _normalize_rows(X)
    ids_arr = np.array(ids, dtype=np.int64)
    return ids_arr, X


def _fetch_color_histograms() -> Tuple[np.ndarray, np.ndarray]:
    """Hole (ids, color_vectors) aus DB, normalisiert (L2)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT image_id, color_histogram
        FROM images
        WHERE color_histogram IS NOT NULL
        ORDER BY image_id ASC
    """)
    rows = cur.fetchall()
    conn.close()

    ids: List[int] = []
    vectors: List[np.ndarray] = []
    for image_id, blob in rows:
        # Erwartete Struktur: ((r,g,b), (h,s,l)) mit z.B. numpy arrays
        (r, g, b), (h, s, l) = pickle.loads(blob)
        vec = np.hstack([r.flatten(), g.flatten(), b.flatten(),
                         h.flatten(), s.flatten(), l.flatten()]).astype(np.float32)
        # Normieren
        nrm = np.linalg.norm(vec)
        if nrm > 0:
            vec /= nrm
        ids.append(int(image_id))
        vectors.append(vec)

    if not vectors:
        raise RuntimeError("No histograms found in DB.")

    X = np.vstack(vectors).astype(np.float32, copy=False)
    ids_arr = np.array(ids, dtype=np.int64)
    return ids_arr, X


# -----------------------------
# IVFPQ Index bauen (mit optional OPQ)
# -----------------------------
def build_ivfpq_index(nlist: int = IVFPQ_DEFAULTS["nlist"],
                      target_code_bytes: int = IVFPQ_DEFAULTS["target_code_bytes"],
                      nprobe: int = IVFPQ_DEFAULTS["nprobe"],
                      use_opq: bool = IVFPQ_DEFAULTS["use_opq"],
                      train_samples: int = IVFPQ_DEFAULTS["train_samples"]) -> None:
    """
    Baue IVFPQ-Index für Embeddings mit optional OPQ.
    :param nlist: Anzahl der Vorverteilungslisten (Cluster)
    :param target_code_bytes: Zielgröße des Codes in Bytes (für m)
    :param nprobe: Anzahl der zu durchsuchenden Listen bei Abfrage
    :param use_opq: Ob OPQ als Vorverarbeitung genutzt werden soll
    :param train_samples: Anzahl der Trainingsbeispiele (für IVFPQ-Training)
    """
    ids, X = _fetch_embeddings()
    N, d = X.shape

    # m so wählen, dass d % m == 0 und grob target_code_bytes entspricht
    m = _choose_m(d, target_code_bytes)

    # Quantizer: IP auf normalisierten Vektoren = Cosine
    quantizer = faiss.IndexFlatIP(d)

    # Optional OPQ (vor IVFPQ als PreTransform)
    if use_opq:
        opq = faiss.OPQMatrix(d, m)  # rotiert die d-Dim so, dass PQ besser arbeitet
        ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 8 Bits/Subquantizer
        index = faiss.IndexPreTransform(opq, ivfpq)
    else:
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

    # Training auf zufälligem Subset (spart RAM/Time)
    rng = np.random.default_rng(RNG_SEED)
    train_count = min(N, int(train_samples))
    train_idx = rng.choice(N, train_count, replace=False)
    train_X = X[train_idx]

    print(f"[IVFPQ] N={N}, d={d}, nlist={nlist}, m={m}, nprobe={nprobe}, OPQ={use_opq}, train={len(train_X)}")
    index.train(train_X)

    # Add mit IDs — bei PreTransform müssen wir an den Wrapper adden
    index.add_with_ids(X, ids)

    # nprobe setzen (bei PreTransform: am inneren Index)
    if isinstance(index, faiss.IndexPreTransform):
        # Inneres IVFPQ greifen
        inner = faiss.downcast_index(index.index)
        inner.nprobe = nprobe
    else:
        index.nprobe = nprobe

    out_path = os.path.join(INDEX_DIR, "emb_ivfpq.faiss")
    faiss.write_index(index, out_path)
    print(f"[IVFPQ] Saved index to {out_path}")

    # Reihenfolge/IDs speichern
    ids_path = os.path.join(INDEX_DIR, "emb_ivfpq.ids.pkl")
    with open(ids_path, "wb") as f:
        pickle.dump(ids, f)
    print(f"[IVFPQ] Saved ID order to {ids_path}")


# -----------------------------
# HNSW Index bauen (Color)
# -----------------------------
def build_hnsw_color_index(M: int = HNSW_DEFAULTS["M"],
                           efConstruction: int = HNSW_DEFAULTS["efConstruction"],
                           efSearch: int = HNSW_DEFAULTS["efSearch"]) -> None:
    """
    Baue HNSW-Index für Color-Histogramme.
    :param M: Anzahl der Nachbarn pro Knoten (Verbindungsgrad)
    :param efConstruction: Effizienzparameter für Konstruktion (höher = besser)
    :param efSearch: Effizienzparameter für Abfragen (höher = besser)
    """
    ids, X = _fetch_color_histograms()
    N, d = X.shape

    # HNSW-Flat (L2 auf normalisierten Vektoren funktioniert gut)
    hnsw = faiss.IndexHNSWFlat(d, M)
    hnsw.hnsw.efConstruction = efConstruction
    # efSearch ist zur Query-Zeit relevant; wir setzen hier einen soliden Default
    hnsw.hnsw.efSearch = efSearch

    # Mit IDs mappen
    index = faiss.IndexIDMap2(hnsw)
    index.add_with_ids(X, ids)

    out_path = os.path.join(INDEX_DIR, "color_hnsw.faiss")
    faiss.write_index(index, out_path)
    print(f"[HNSW] Saved index with {len(ids)} vectors to {out_path} (M={M}, efC={efConstruction}, efS={efSearch})")

    ids_path = os.path.join(INDEX_DIR, "color_hnsw.ids.pkl")
    with open(ids_path, "wb") as f:
        pickle.dump(ids, f)
    print(f"[HNSW] Saved ID order to {ids_path}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    build_ivfpq_index(
        nlist=IVFPQ_DEFAULTS["nlist"],
        target_code_bytes=IVFPQ_DEFAULTS["target_code_bytes"],
        nprobe=IVFPQ_DEFAULTS["nprobe"],
        use_opq=IVFPQ_DEFAULTS["use_opq"],
        train_samples=IVFPQ_DEFAULTS["train_samples"],
    )
    build_hnsw_color_index(
        M=HNSW_DEFAULTS["M"],
        efConstruction=HNSW_DEFAULTS["efConstruction"],
        efSearch=HNSW_DEFAULTS["efSearch"],
    )
