import os
import pickle
import shutil
import sqlite3
import tempfile

import faiss
import numpy as np


DB_PATH = "images_database.db"
INDEX_DIR = "indexes"


# ----------------- Database helpers -----------------

def iter_embeddings(conn):
    """
    Stream embeddings from the database in deterministic image_id order.
    """

    cur = conn.cursor()

    cur.execute(
        """
        SELECT image_id, embedding
        FROM images
        WHERE embedding IS NOT NULL
        ORDER BY image_id ASC
        """
    )

    while True:
        rows = cur.fetchmany(5000)

        if not rows:
            break

        for image_id, blob in rows:
            yield image_id, blob


def iter_color_histograms(conn):
    """
    Stream color histograms from the database in deterministic
    image_id order.
    """

    cur = conn.cursor()

    cur.execute(
        """
        SELECT image_id, color_histogram
        FROM images
        WHERE color_histogram IS NOT NULL
        ORDER BY image_id ASC
        """
    )

    while True:
        rows = cur.fetchmany(5000)

        if not rows:
            break

        for image_id, blob in rows:
            yield image_id, blob


# ----------------- IVFPQ -----------------

def build_ivfpq_index(output_dir):
    """
    Build an IVFPQ index for image embeddings.

    The index and its corresponding ID file are written into
    output_dir. Nothing is written to the production index directory
    during the build.
    """

    conn = sqlite3.connect(DB_PATH)

    ids = []
    vectors = []

    try:
        for image_id, blob in iter_embeddings(conn):
            vector = pickle.loads(blob).astype(
                np.float32,
                copy=False,
            )

            norm = np.linalg.norm(vector)

            if norm > 0:
                vector = vector / norm

            ids.append(int(image_id))
            vectors.append(vector)

    finally:
        conn.close()

    if not vectors:
        raise RuntimeError(
            "No embeddings found in database."
        )

    x = np.vstack(vectors).astype(
        np.float32,
        copy=False,
    )

    ids = np.asarray(
        ids,
        dtype=np.int64,
    )

    d = x.shape[1]
    n = len(vectors)

    # Dynamically determine the number of IVF clusters.
    nlist = max(
        64,
        min(
            int(n ** 0.5),
            n // 4,
        ),
    )

    # PQ requires the vector dimension to be divisible by m.
    m = 16

    if d % m != 0:
        raise RuntimeError(
            f"Embedding dimension {d} is not divisible "
            f"by m={m}. Please choose a suitable PQ segment count."
        )

    quantizer = faiss.IndexFlatIP(d)

    index = faiss.IndexIVFPQ(
        quantizer,
        d,
        nlist,
        m,
        8,
    )

    print(
        f"[IVFPQ] Training with {n} vectors, "
        f"dim={d}, nlist={nlist}, m={m}"
    )

    index.train(x)

    if not index.is_trained:
        raise RuntimeError(
            "IVFPQ index training failed."
        )

    index.add_with_ids(
        x,
        ids,
    )

    index.nprobe = 16

    index_path = os.path.join(
        output_dir,
        "emb_ivfpq.faiss",
    )

    faiss.write_index(
        index,
        index_path,
    )

    print(
        f"[IVFPQ] Saved index to {index_path}"
    )

    # Save the corresponding image IDs.
    ids_path = os.path.join(
        output_dir,
        "emb_ivfpq.ids.pkl",
    )

    with open(ids_path, "wb") as f:
        pickle.dump(
            ids,
            f,
        )

    print(
        f"[IVFPQ] Saved ID order to {ids_path}"
    )


# ----------------- HNSW Color -----------------

def build_hnsw_color_index(output_dir):
    """
    Build an HNSW index for color histograms.

    The index and its corresponding ID file are written into
    output_dir. Nothing is written to the production index directory
    during the build.
    """

    conn = sqlite3.connect(DB_PATH)

    ids = []
    vectors = []

    try:
        for image_id, blob in iter_color_histograms(conn):
            # Expected structure:
            # ((r, g, b), (h, s, l))
            (
                (r, g, b),
                (h, s, l),
            ) = pickle.loads(blob)

            vector = np.hstack(
                [
                    r.flatten(),
                    g.flatten(),
                    b.flatten(),
                    h.flatten(),
                    s.flatten(),
                    l.flatten(),
                ]
            ).astype(
                np.float32,
                copy=False,
            )

            norm = np.linalg.norm(vector)

            if norm > 0:
                vector = vector / norm

            ids.append(int(image_id))
            vectors.append(vector)

    finally:
        conn.close()

    if not vectors:
        raise RuntimeError(
            "No color histograms found in database."
        )

    x = np.vstack(vectors).astype(
        np.float32,
        copy=False,
    )

    ids = np.asarray(
        ids,
        dtype=np.int64,
    )

    d = x.shape[1]

    # Build HNSW index.
    hnsw = faiss.IndexHNSWFlat(
        d,
        32,
    )

    hnsw.hnsw.efConstruction = 4096

    index = faiss.IndexIDMap2(
        hnsw
    )

    index.add_with_ids(
        x,
        ids,
    )

    index_path = os.path.join(
        output_dir,
        "color_hnsw.faiss",
    )

    faiss.write_index(
        index,
        index_path,
    )

    print(
        f"[HNSW] Saved index with "
        f"{len(ids)} vectors to {index_path}"
    )

    # Save the corresponding image IDs.
    ids_path = os.path.join(
        output_dir,
        "color_hnsw.ids.pkl",
    )

    with open(ids_path, "wb") as f:
        pickle.dump(
            ids,
            f,
        )

    print(
        f"[HNSW] Saved ID order to {ids_path}"
    )


# ----------------- Full FAISS setup -----------------

def build_faiss_indexes():
    """
    Build all FAISS indexes in a temporary directory.

    The existing production indexes remain untouched until every
    index has been built successfully.

    Returns:
        True  -> all indexes were built and installed successfully
        False -> build failed and existing indexes were preserved
    """

    index_dir = os.path.abspath(
        INDEX_DIR
    )

    parent_dir = os.path.dirname(
        index_dir
    )

    os.makedirs(
        parent_dir,
        exist_ok=True,
    )

    temp_index_dir = tempfile.mkdtemp(
        prefix=".indexes_",
        dir=parent_dir,
    )

    print(
        f"[INFO] Building FAISS indexes in "
        f"temporary directory: {temp_index_dir}"
    )

    try:
        # -----------------------------------------------------
        # Build all indexes in the temporary directory.
        # -----------------------------------------------------

        build_ivfpq_index(
            temp_index_dir
        )

        build_hnsw_color_index(
            temp_index_dir
        )

        # -----------------------------------------------------
        # Verify that all expected files exist before replacing
        # the production indexes.
        # -----------------------------------------------------

        expected_files = [
            "emb_ivfpq.faiss",
            "emb_ivfpq.ids.pkl",
            "color_hnsw.faiss",
            "color_hnsw.ids.pkl",
        ]

        for filename in expected_files:
            path = os.path.join(
                temp_index_dir,
                filename,
            )

            if not os.path.isfile(path):
                raise RuntimeError(
                    f"Expected FAISS output is missing: "
                    f"{path}"
                )

        # -----------------------------------------------------
        # Verify that the generated indexes can actually be
        # opened again before installing them.
        # -----------------------------------------------------

        print(
            "[INFO] Verifying generated FAISS indexes..."
        )

        emb_index = faiss.read_index(
            os.path.join(
                temp_index_dir,
                "emb_ivfpq.faiss",
            )
        )

        color_index = faiss.read_index(
            os.path.join(
                temp_index_dir,
                "color_hnsw.faiss",
            )
        )

        if emb_index.ntotal == 0:
            raise RuntimeError(
                "Generated embedding FAISS index is empty."
            )

        if color_index.ntotal == 0:
            raise RuntimeError(
                "Generated color FAISS index is empty."
            )

        print(
            f"[OK] Embedding index contains "
            f"{emb_index.ntotal} vectors."
        )

        print(
            f"[OK] Color index contains "
            f"{color_index.ntotal} vectors."
        )

        # -----------------------------------------------------
        # Close Python references before replacing the directory.
        # -----------------------------------------------------

        del emb_index
        del color_index

        # -----------------------------------------------------
        # Install the new index directory.
        #
        # Keep the old directory until the new one is ready.
        # -----------------------------------------------------

        old_index_dir = None

        if os.path.exists(index_dir):
            old_index_dir = tempfile.mkdtemp(
                prefix=".indexes_old_",
                dir=parent_dir,
            )

            # Remove the empty temporary directory first.
            os.rmdir(old_index_dir)

            print(
                "[INFO] Moving existing FAISS indexes "
                "to temporary backup location..."
            )

            os.replace(
                index_dir,
                old_index_dir,
            )

        try:
            print(
                "[INFO] Installing new FAISS indexes..."
            )

            os.replace(
                temp_index_dir,
                index_dir,
            )

            temp_index_dir = None

        except Exception:
            # If installation fails, restore the old index
            # directory if it was moved away.
            if (
                old_index_dir is not None
                and os.path.exists(old_index_dir)
                and not os.path.exists(index_dir)
            ):
                os.replace(
                    old_index_dir,
                    index_dir,
                )

            raise

        # -----------------------------------------------------
        # The new indexes are now installed successfully.
        # The old indexes are no longer needed.
        # -----------------------------------------------------

        if (
            old_index_dir is not None
            and os.path.exists(old_index_dir)
        ):
            shutil.rmtree(
                old_index_dir
            )

        print(
            "[DONE] All FAISS indexes rebuilt "
            "and installed successfully."
        )

        return True

    except Exception as e:
        print(
            f"[ERROR] FAISS setup failed: {e}"
        )

        print(
            "[INFO] Existing FAISS indexes were preserved."
        )

        return False

    finally:
        # -----------------------------------------------------
        # Remove incomplete temporary index directory.
        # -----------------------------------------------------

        if (
            temp_index_dir is not None
            and os.path.exists(temp_index_dir)
        ):
            try:
                shutil.rmtree(
                    temp_index_dir
                )

                print(
                    "[INFO] Temporary FAISS directory removed."
                )

            except OSError as e:
                print(
                    f"[WARN] Could not remove temporary "
                    f"FAISS directory: {e}"
                )


# ----------------- Example -----------------

if __name__ == "__main__":

    success = build_faiss_indexes()

    if success:
        print(
            "[OK] FAISS setup finished successfully."
        )
    else:
        print(
            "[ERROR] FAISS setup failed."
        )
