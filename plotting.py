#!/usr/bin/env python3
import argparse
import os
import pickle
import sqlite3
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Optional deps
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


def log(msg: str):
    print(msg, flush=True)


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def reduce_dense(X: np.ndarray, method: str, seed: int, n_components: int = 2) -> np.ndarray:
    method = (method or "umap").lower()
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=seed, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(X)
    if method == "tsne":
        return TSNE(n_components=n_components, random_state=seed, init="pca", learning_rate="auto", perplexity=30.0).fit_transform(X)
    if method == "pca":
        return PCA(n_components=n_components, random_state=seed).fit_transform(X)
    # Fallbacks
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=seed, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(X)
    try:
        return TSNE(n_components=n_components, random_state=seed, init="pca", learning_rate="auto").fit_transform(X)
    except Exception:
        return PCA(n_components=n_components).fit_transform(X)


def scatter_save(X2: np.ndarray, ids: List[str], title: str, out_png: str, out_csv: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1], s=6, alpha=0.8)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    header = "id,x,y"
    data = np.column_stack([np.array(ids, dtype=object), X2])
    np.savetxt(out_csv, data, delimiter=",", fmt="%s", header=header, comments="")


def load_pickle_like(path: str):
    if path.lower().endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        if "ids" in data and "vectors" in data:
            ids = list(map(str, data["ids"].tolist()))
            values = np.asarray(data["vectors"])
            return ids, values
        arr = np.asarray(data[list(data.files)[0]])
        ids = [str(i) for i in range(arr.shape[0])]
        return ids, arr
    if path.lower().endswith(".npy"):
        obj = np.load(path, allow_pickle=True)
    else:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    if isinstance(obj, dict):
        ids = list(map(str, obj.keys()))
        vals = list(obj.values())
        try:
            arr = np.vstack([np.asarray(v).ravel() for v in vals])
            return ids, arr
        except Exception:
            return ids, np.array(vals, dtype=object)
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        ids, vals = obj
        return list(map(str, ids)), np.asarray(vals)
    arr = np.asarray(obj)
    return [str(i) for i in range(arr.shape[0])], arr


def hash_to_bits_any(obj) -> Optional[np.ndarray]:
    try:
        if hasattr(obj, "hash"):
            return np.asarray(obj.hash, dtype=np.uint8).ravel()
        arr = np.asarray(obj)
        if arr.dtype == np.bool_:
            return arr.astype(np.uint8).ravel()
        if np.issubdtype(arr.dtype, np.integer):
            val = int(arr)
            bits = np.unpackbits(np.array([val], dtype=">u8").view(np.uint8))
            return bits.astype(np.uint8)
        return arr.astype(np.uint8).ravel()
    except Exception:
        return None


def plot_dense(ids: List[str], X: np.ndarray, title: str, method: str, out_png: str, out_csv: str, seed: int) -> bool:
    if len(ids) < 3 or X is None or X.size == 0:
        return False
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    X2 = reduce_dense(Xn, method, seed)
    scatter_save(X2, ids, title, out_png, out_csv)
    return True


def plot_hash(ids: List[str], values: Iterable, out_png: str, out_csv: str, seed: int) -> bool:
    bits_all = []
    for v in values:
        b = hash_to_bits_any(v)
        if b is not None:
            bits_all.append(b)
    if len(bits_all) < 3:
        return False
    max_len = max(len(b) for b in bits_all)
    B = np.zeros((len(bits_all), max_len), dtype=np.uint8)
    for i, b in enumerate(bits_all):
        L = min(len(b), max_len)
        B[i, :L] = b[:L]
    D = pairwise_distances(B, metric="hamming")
    try:
        X2 = TSNE(n_components=2, metric="precomputed", random_state=seed, init="random", learning_rate="auto").fit_transform(D)
    except Exception:
        D2 = D ** 2
        n = D2.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K = -0.5 * H @ D2 @ H
        X2 = PCA(n_components=2, random_state=seed).fit_transform(K)
    scatter_save(X2, ids, "Hash similarity (Hamming + t-SNE)", out_png, out_csv)
    return True


def sqlite_fetch(db_path: str, cols: List[str], limit: Optional[int], query: Optional[str]) -> List[Tuple]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        if query:
            q = query
        else:
            q = f"SELECT {', '.join(cols)} FROM images"
            if limit is not None:
                q += f" LIMIT {int(limit)}"
        cur.execute(q)
        return cur.fetchall()
    finally:
        conn.close()


def load_faiss_vectors(index_path: str, ids_path: Optional[str]):
    if not HAS_FAISS:
        raise RuntimeError("faiss not available.")
    index = faiss.read_index(index_path)
    ntotal = index.ntotal
    # load ids
    if ids_path:
        if ids_path.lower().endswith(".npy"):
            arr = np.load(ids_path, allow_pickle=True)
            ids = list(map(str, arr.tolist()))
        else:
            with open(ids_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "ids" in obj:
                ids = list(map(str, obj["ids"]))
            elif isinstance(obj, (list, tuple, np.ndarray)):
                ids = list(map(str, list(obj)))
            else:
                ids = [str(obj)]
    else:
        ids = [str(i) for i in range(ntotal)]
    # reconstruct
    vecs = []
    can_reconstruct = hasattr(index, "reconstruct")
    xb = getattr(index, "xb", None)
    if can_reconstruct:
        for i in range(ntotal):
            try:
                v = index.reconstruct(i)
                vecs.append(np.asarray(v, dtype=np.float32))
            except Exception:
                break
    if len(vecs) != ntotal and xb is not None:
        arr = np.array(xb, dtype=np.float32)
        if arr.shape[0] >= ntotal:
            vecs = [arr[i] for i in range(ntotal)]
    if len(vecs) != ntotal:
        raise RuntimeError("Could not reconstruct vectors from FAISS index.")
    X = np.vstack(vecs).astype(np.float32)
    if len(ids) != ntotal:
        ids = ids[:ntotal]
    return ids, X


def main():
    log(">> plotting.py starting up")
    ap = argparse.ArgumentParser(description="Plot 2D layouts from SQLite, Pickle/NPZ/NPY, or FAISS sources.")
    ap.add_argument("--outdir", type=str, default="plots", help="Output directory for PNGs/CSVs.")
    ap.add_argument("--method", type=str, default="umap", choices=["umap", "tsne", "pca"], help="Reducer for dense features.")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="Optional subsample for SQLite.")
    # SQLite
    ap.add_argument("--db", type=str, help="SQLite DB path.")
    ap.add_argument("--color-query", type=str, default=None)
    ap.add_argument("--embedding-query", type=str, default=None)
    ap.add_argument("--hashing-query", type=str, default=None)
    # Pickle
    ap.add_argument("--color-pkl", type=str)
    ap.add_argument("--embedding-pkl", type=str)
    ap.add_argument("--hashing-pkl", type=str)
    # FAISS
    ap.add_argument("--faiss-embedding", type=str)
    ap.add_argument("--faiss-ids", type=str)
    ap.add_argument("--faiss-color", type=str)
    ap.add_argument("--faiss-color-ids", type=str)
    args = ap.parse_args()

    log(f">> args: {args}")
    ensure_outdir(args.outdir)
    log(f">> writing outputs to: {args.outdir}")

    color_done = False
    embedding_done = False
    hashing_done = False

    # Color: pickle
    if args.color_pkl:
        log("Color: trying pickle source...")
        ids, vecs_or_obj = load_pickle_like(args.color_pkl)
        if isinstance(vecs_or_obj, np.ndarray) and vecs_or_obj.dtype != object:
            color_done = plot_dense(
                ids, vecs_or_obj, f"Color similarity ({args.method.upper()})",
                args.method, os.path.join(args.outdir, "color_layout.png"),
                os.path.join(args.outdir, "color_layout.csv"),
                args.random_state
            )
        log("[OK] Color (pickle)" if color_done else "[SKIP] Color (pickle)")

    # Color: FAISS
    if not color_done and args.faiss_color:
        log("Color: trying FAISS source...")
        try:
            ids_c, vecs_c = load_faiss_vectors(args.faiss_color, args.faiss_color_ids)
            color_done = plot_dense(
                ids_c, vecs_c, f"Color similarity ({args.method.upper()})",
                args.method, os.path.join(args.outdir, "color_layout.png"),
                os.path.join(args.outdir, "color_layout.csv"),
                args.random_state
            )
        except Exception as e:
            log(f"[Color/FAISS] skipped: {e}")
        log("[OK] Color (faiss)" if color_done else "[SKIP] Color (faiss)")

    # Color: DB
    if not color_done and args.db:
        log("Color: trying DB source...")
        try:
            rows = sqlite_fetch(args.db, ["image_id", "color_histogram"], args.limit, args.color_query)
            ids, vecs = [], []
            for img_id, blob in rows:
                if blob is None:
                    continue
                try:
                    val = pickle.loads(blob)
                except Exception:
                    continue
                try:
                    rgb, hsl = val
                    arrs = list(rgb) + list(hsl)
                    flat = np.concatenate([np.asarray(a).ravel() for a in arrs]).astype(np.float32)
                    s = flat.sum()
                    if s > 0:
                        flat = flat / s
                except Exception:
                    try:
                        flat = np.asarray(val).astype(np.float32).ravel()
                        s = flat.sum()
                        if s > 0:
                            flat = flat / s
                    except Exception:
                        continue
                ids.append(str(img_id))
                vecs.append(flat)
            if vecs:
                X = np.vstack(vecs)
                color_done = plot_dense(
                    ids, X, f"Color similarity ({args.method.upper()})",
                    args.method, os.path.join(args.outdir, "color_layout.png"),
                    os.path.join(args.outdir, "color_layout.csv"),
                    args.random_state
                )
        except Exception as e:
            log(f"[Color/DB] skipped: {e}")
        log("[OK] Color (db)" if color_done else "[SKIP] Color (db)")

    # Embedding: pickle
    if args.embedding_pkl:
        log("Embedding: trying pickle source...")
        ids, vecs = load_pickle_like(args.embedding_pkl)
        if isinstance(vecs, np.ndarray) and vecs.dtype != object:
            embedding_done = plot_dense(
                ids, vecs, f"Embedding similarity ({args.method.upper()})",
                args.method, os.path.join(args.outdir, "embedding_layout.png"),
                os.path.join(args.outdir, "embedding_layout.csv"),
                args.random_state
            )
        log("[OK] Embedding (pickle)" if embedding_done else "[SKIP] Embedding (pickle)")

    # Embedding: FAISS
    if not embedding_done and args.faiss_embedding:
        log("Embedding: trying FAISS source...")
        try:
            ids_e, vecs_e = load_faiss_vectors(args.faiss_embedding, args.faiss_ids)
            embedding_done = plot_dense(
                ids_e, vecs_e, f"Embedding similarity ({args.method.upper()})",
                args.method, os.path.join(args.outdir, "embedding_layout.png"),
                os.path.join(args.outdir, "embedding_layout.csv"),
                args.random_state
            )
        except Exception as e:
            log(f"[Embedding/FAISS] skipped: {e}")
        log("[OK] Embedding (faiss)" if embedding_done else "[SKIP] Embedding (faiss)")

    # Embedding: DB
    if not embedding_done and args.db:
        log("Embedding: trying DB source...")
        try:
            rows = sqlite_fetch(args.db, ["image_id", "embedding"], args.limit, args.embedding_query)
            ids, vecs = [], []
            for img_id, blob in rows:
                if blob is None:
                    continue
                try:
                    v = pickle.loads(blob)
                    arr = np.asarray(v).astype(np.float32).ravel()
                except Exception:
                    continue
                ids.append(str(img_id))
                vecs.append(arr)
            if vecs:
                X = np.vstack(vecs)
                embedding_done = plot_dense(
                    ids, X, f"Embedding similarity ({args.method.upper()})",
                    args.method, os.path.join(args.outdir, "embedding_layout.png"),
                    os.path.join(args.outdir, "embedding_layout.csv"),
                    args.random_state
                )
        except Exception as e:
            log(f"[Embedding/DB] skipped: {e}")
        log("[OK] Embedding (db)" if embedding_done else "[SKIP] Embedding (db)")

    # Hashing: pickle
    if args.hashing_pkl:
        log("Hashing: trying pickle source...")
        ids, values = load_pickle_like(args.hashing_pkl)
        hashing_done = plot_hash(
            ids, values, os.path.join(args.outdir, "hashing_layout.png"),
            os.path.join(args.outdir, "hashing_layout.csv"),
            args.random_state
        )
        log("[OK] Hashing (pickle)" if hashing_done else "[SKIP] Hashing (pickle)")

    # Hashing: DB
    if not hashing_done and args.db:
        log("Hashing: trying DB source...")
        try:
            rows = sqlite_fetch(args.db, ["image_id", "image_hash"], args.limit, args.hashing_query)
            ids, vals = [], []
            for img_id, blob in rows:
                if blob is None:
                    continue
                try:
                    val = pickle.loads(blob)
                except Exception:
                    val = blob
                ids.append(str(img_id))
                vals.append(val)
            if ids:
                hashing_done = plot_hash(
                    ids, vals, os.path.join(args.outdir, "hashing_layout.png"),
                    os.path.join(args.outdir, "hashing_layout.csv"),
                    args.random_state
                )
        except Exception as e:
            log(f"[Hashing/DB] skipped: {e}")
        log("[OK] Hashing (db)" if hashing_done else "[SKIP] Hashing (db)")

    any_ok = color_done or embedding_done or hashing_done
    if not any_ok:
        log("No plots produced. Provide at least one valid source for any metric.")
        raise SystemExit(1)
    else:
        log("Done. PNGs and CSVs saved to " + args.outdir)


if __name__ == "__main__":
    main()
