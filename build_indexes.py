import sqlite3
import pickle
import numpy as np
import faiss
import os

DB_PATH = "images.db"
INDEX_DIR = "indexes"

os.makedirs(INDEX_DIR, exist_ok=True)


def build_ivfpq_index():
    """Builds an IVFPQ index for embeddings."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT image_id, embedding FROM images WHERE embedding IS NOT NULL;")
    rows = cur.fetchall()
    conn.close()

    ids, vecs = [], []
    for image_id, blob in rows:
        v = pickle.loads(blob).astype(np.float32, copy=False)
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        ids.append(image_id)
        vecs.append(v)

    if not vecs:
        raise RuntimeError("No embeddings found in DB.")

    x = np.vstack(vecs).astype(np.float32, copy=False)
    ids = np.array(ids, dtype=np.int64)
    d = x.shape[1]

    N = len(vecs)
    nlist = max(64, min(int(N ** 0.5), N // 4))  # dynamic cluster count
    m = 16  # PQ segments

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

    print(f"[IVFPQ] Training with {N} vectors, dim={d}, nlist={nlist}, m={m}")
    index.train(x)
    index.add_with_ids(x, ids)
    index.nprobe = 16

    out_path = os.path.join(INDEX_DIR, "emb_ivfpq.faiss")
    faiss.write_index(index, out_path)
    print(f"[IVFPQ] Saved index to {out_path}")


def build_hnsw_color_index():
    DB_PATH = "images.db"
    INDEX_DIR = "indexes"
    os.makedirs(INDEX_DIR, exist_ok=True)
    OUT_PATH = os.path.join(INDEX_DIR, "color_hnsw.faiss")

    # Load color histograms from DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT image_id, color_histogram FROM images WHERE color_histogram IS NOT NULL;")
    rows = cur.fetchall()
    conn.close()

    ids = []
    vectors = []
    for image_id, blob in rows:
        # Unpickle histogram tuple ((r,g,b), (h,s,l))
        (r, g, b), (h, s, l) = pickle.loads(blob)
        # Flatten and concatenate into one vector
        vec = np.hstack([
            r.flatten(), g.flatten(), b.flatten(),
            h.flatten(), s.flatten(), l.flatten()
        ]).astype(np.float32)

        # Normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        ids.append(int(image_id))
        vectors.append(vec)

    X = np.vstack(vectors).astype(np.float32)
    ids = np.array(ids, dtype=np.int64)

    # Build FAISS HNSW index + wrap with ID map so we can use image_ids
    d = X.shape[1]
    hnsw = faiss.IndexHNSWFlat(d, 32)     # HNSW with M=32
    hnsw.hnsw.efConstruction = 4096

    # Wrap to support add_with_ids
    index = faiss.IndexIDMap2(hnsw)
    index.add_with_ids(X, ids)

    # Save index
    faiss.write_index(index, OUT_PATH)
    print(f"[FAISS-HNSW] Saved index with {len(ids)} vectors to {OUT_PATH}")



if __name__ == "__main__":
    build_ivfpq_index()
    build_hnsw_color_index()
