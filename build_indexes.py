import sqlite3
import pickle
import numpy as np
import faiss
import os

DB_PATH = "images_database.db"
INDEX_DIR = "indexes"
os.makedirs(INDEX_DIR, exist_ok=True)


def build_ivfpq_index():
    """Build an IVFPQ index for embeddings and save ID order."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # embeddings holen, sortiert nach image_id (deterministisch!)
    cur.execute("""
        SELECT image_id, embedding
        FROM images
        WHERE embedding IS NOT NULL
        ORDER BY image_id ASC
    """)
    rows = cur.fetchall()
    conn.close()

    ids, vecs = [], []
    for image_id, blob in rows:
        v = pickle.loads(blob).astype(np.float32, copy=False)
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        ids.append(int(image_id))
        vecs.append(v)

    if not vecs:
        raise RuntimeError("No embeddings found in DB.")

    x = np.vstack(vecs).astype(np.float32, copy=False)
    ids = np.array(ids, dtype=np.int64)
    d = x.shape[1]

    N = len(vecs)
    nlist = max(64, min(int(N ** 0.5), N // 4))  # Cluster dynamisch
    m = 16  # PQ-Segmente

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

    print(f"[IVFPQ] Training with {N} vectors, dim={d}, nlist={nlist}, m={m}")
    index.train(x)
    index.add_with_ids(x, ids)
    index.nprobe = 16

    out_path = os.path.join(INDEX_DIR, "emb_ivfpq.faiss")
    faiss.write_index(index, out_path)
    print(f"[IVFPQ] Saved index to {out_path}")

    # Reihenfolge der IDs speichern
    ids_path = os.path.join(INDEX_DIR, "emb_ivfpq.ids.pkl")
    with open(ids_path, "wb") as f:
        pickle.dump(ids, f)
    print(f"[IVFPQ] Saved ID order to {ids_path}")


def build_hnsw_color_index():
    """Build an HNSW index for color histograms and save ID order."""
    OUT_PATH = os.path.join(INDEX_DIR, "color_hnsw.faiss")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Farbhistogramme holen, sortiert nach image_id
    cur.execute("""
        SELECT image_id, color_histogram
        FROM images
        WHERE color_histogram IS NOT NULL
        ORDER BY image_id ASC
    """)
    rows = cur.fetchall()
    conn.close()

    ids, vectors = [], []
    for image_id, blob in rows:
        # Beispiel: Unpickle ((r,g,b), (h,s,l))
        (r, g, b), (h, s, l) = pickle.loads(blob)
        vec = np.hstack([
            r.flatten(), g.flatten(), b.flatten(),
            h.flatten(), s.flatten(), l.flatten()
        ]).astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        ids.append(int(image_id))
        vectors.append(vec)

    if not vectors:
        raise RuntimeError("No histograms found in DB.")

    X = np.vstack(vectors).astype(np.float32)
    ids = np.array(ids, dtype=np.int64)
    d = X.shape[1]

    # HNSW bauen
    hnsw = faiss.IndexHNSWFlat(d, 32)
    hnsw.hnsw.efConstruction = 4096
    index = faiss.IndexIDMap2(hnsw)
    index.add_with_ids(X, ids)

    faiss.write_index(index, OUT_PATH)
    print(f"[HNSW] Saved index with {len(ids)} vectors to {OUT_PATH}")

    # Reihenfolge der IDs speichern
    ids_path = os.path.join(INDEX_DIR, "color_hnsw.ids.pkl")
    with open(ids_path, "wb") as f:
        pickle.dump(ids, f)
    print(f"[HNSW] Saved ID order to {ids_path}")


if __name__ == "__main__":
    build_ivfpq_index()
    build_hnsw_color_index()
