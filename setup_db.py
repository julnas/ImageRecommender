import os
import pickle
import sqlite3
import numpy as np
from typing import Optional
from PIL import Image, UnidentifiedImageError, ImageFile

# erlaubt das Laden „komischer“/teilweise defekter Dateien statt hart zu crashen
ImageFile.LOAD_TRUNCATED_IMAGES = True

from image_recommender.database import Database
from image_recommender.image_loader import ImageLoader
from SimilarityMetrics.color_similarity_metric import ColorSimilarity
from SimilarityMetrics.embeddings_similarity_metric import EmbeddingSimilarity
from SimilarityMetrics.hashing_similarity_metric import HashingSimilarity


# ----------------- Robuster Generator für Bildpfade -----------------
ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}  # bei Bedarf erweitern: ".tif", ".tiff", ".heic"

def iter_image_paths(
    base_dir: str,
    follow_links: bool = False,
    exclude: Optional[list[str]] = None,
):
    """
    Liefert absolute Pfade zu Bilddateien unter base_dir (rekursiv).

    exclude: Liste von Einträgen, die ausgelassen werden sollen.
      - absoluter Ordnerpfad: "/Volumes/.../FFHQ_images/23000"
      - relativer Pfad ab base_dir: "image_data/FFHQ_images/23000"
      - nur Ordnername: "thumbnails" (wird überall ignoriert)
    """
    exclude = exclude or []

    abs_exclude_paths, rel_exclude_paths, exclude_names = set(), set(), set()
    base_dir_abs = os.path.normpath(os.path.abspath(base_dir))

    for item in exclude:
        if os.path.isabs(item):
            p = os.path.normpath(item)
            abs_exclude_paths.add(p)
            exclude_names.add(os.path.basename(p))
        else:
            rel = os.path.normpath(item)
            rel_exclude_paths.add(rel)
            abs_exclude_paths.add(os.path.normpath(os.path.join(base_dir_abs, rel)))
            exclude_names.add(os.path.basename(rel))

    for root, dirs, files in os.walk(base_dir_abs, followlinks=follow_links):
        root_abs = os.path.normpath(root)

        # Unterordner in-place filtern -> nicht betreten
        kept = []
        for d in dirs:
            child_abs = os.path.normpath(os.path.join(root_abs, d))
            child_rel = os.path.normpath(os.path.relpath(child_abs, base_dir_abs))
            if child_abs in abs_exclude_paths or child_rel in rel_exclude_paths or d in exclude_names:
                continue
            kept.append(d)
        dirs[:] = kept

        for fname in files:
            if fname.startswith("._"):
                continue
            if os.path.splitext(fname)[1].lower() in ALLOWED_EXTS:
                yield os.path.join(root_abs, fname)



def scan_and_fill_database(
    base_dir: str,
    db_path: str = "images_database.db",
    max_images: Optional[int] = None,
    commit_batch_size: int = 1000
):
    """
    Scans `base_dir` recursively for images and stores features in SQLite.
    Uses a streaming iterator + batch commits (default: every 1000 images).
    Skips unreadable files and never accumulates features in RAM.
    """

    # --- DB-Verbindung ---
    db = Database(db_path)
    # robustere SQLite-Einstellungen für lange Läufe
    db.cursor.execute("PRAGMA journal_mode=WAL;")
    db.cursor.execute("PRAGMA synchronous=NORMAL;")
    db.cursor.execute("PRAGMA busy_timeout=10000;")
    db.connection.commit()

    db.cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL UNIQUE,
            color_histogram BLOB,
            embedding BLOB,
            image_hash BLOB,
            width INTEGER,
            height INTEGER,
            file_size INTEGER
        );
        """
    )
    db.connection.commit()

    # --- Loader & Metriken ---
    loader = ImageLoader(db, base_dir)
    color_similarity = ColorSimilarity(loader)
    embedding_similarity = EmbeddingSimilarity(loader)
    hashing_similarity = HashingSimilarity(loader)

    count = 0
    batch_count = 0
    skipped = 0

    # --- Rekursiv durch alle Bilder ---
    for full_path in iter_image_paths(
    base_dir=base_dir, 
    follow_links=False, # Links nicht folgen, um z.B. Symlinks zu vermeiden
    exclude=[], # Möglichkeit zum Ausschließen von Ordnern
    ):
        if max_images and count >= max_images:
            print(f"[INFO] Stopped after {max_images} images.")
            break

        relative_path = os.path.relpath(full_path, base_dir)

        # Bild laden (sicher, Filehandle direkt schließen)
        try:
            img = loader.load_image_by_path(full_path)
            if img is None:
                with Image.open(full_path) as im:
                    img = im.convert("RGB").copy()
            else:
                img = img.convert("RGB").copy()
        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARN] Skip (cannot open): {relative_path} -> {e}")
            skipped += 1
            continue
        except Exception as e:
            print(f"[WARN] Skip (open exception): {relative_path} -> {e}")
            skipped += 1
            continue

        # Features berechnen
        try:
            color_feature = color_similarity.compute_feature(img)
            embedding_vector = embedding_similarity.compute_feature(img)
            hash_value = hashing_similarity.compute_feature(img)
        except Exception as e:
            print(f"[WARN] Skip (feature error): {relative_path} -> {e}")
            skipped += 1
            continue

        # Serialisieren für DB
        color_blob = pickle.dumps(color_feature)
        embedding_blob = pickle.dumps(embedding_vector)
        hash_blob = pickle.dumps(hash_value)

        width, height = img.size
        file_size = os.path.getsize(full_path) if os.path.exists(full_path) else None

        try:
            db.cursor.execute(
                """
                INSERT INTO images
                    (file_path, color_histogram, embedding, image_hash, width, height, file_size)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    color_histogram = excluded.color_histogram,
                    embedding = excluded.embedding,
                    image_hash = excluded.image_hash,
                    width = excluded.width,
                    height = excluded.height,
                    file_size = excluded.file_size;
                """,
                (
                    relative_path,
                    color_blob,
                    embedding_blob,
                    hash_blob,
                    width,
                    height,
                    file_size,
                ),
            )
        except sqlite3.Error as e:
            print(f"[WARN] DB write failed for {relative_path}: {e}")
            skipped += 1
            continue

        count += 1
        batch_count += 1

        if batch_count >= commit_batch_size:
            db.connection.commit()
            print(f"[OK] Committed batch of {batch_count} (total processed: {count}, skipped: {skipped})")
            batch_count = 0

        if count % 200 == 0:
            # Mini-Heartbeat ohne Commit (optional)
            print(f"[INFO] Processed so far: {count} (skipped: {skipped})")

    # Letzten Batch committen
    if batch_count > 0:
        db.connection.commit()
        print(f"[OK] Final commit of {batch_count} (total processed: {count}, skipped: {skipped})")

    db.close()

    print(f"[DONE] Database updated. Processed: {count}, skipped: {skipped}.")


if __name__ == "__main__":
    # Beispielaufruf: Scan und Füllen der Datenbank
    base_dir = "/Volumes/BigData03/data"
    scan_and_fill_database(base_dir, max_images=None, commit_batch_size=5000)
