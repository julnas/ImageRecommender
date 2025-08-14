import os
import pickle
import numpy as np
from typing import Optional
from PIL import Image, UnidentifiedImageError

from image_recommender.database import Database
from SimilarityMetrics.color_similarity_metric import ColorSimilarity
from SimilarityMetrics.embeddings_similarity_metric import EmbeddingSimilarity
from SimilarityMetrics.hashing_similarity_metric import HashingSimilarity
from image_recommender.image_loader import ImageLoader


def scan_and_fill_database(base_dir: str, db_path: str = "images.db", max_images: Optional[int] = None):
    """
    Scans base_dir for images and inserts/updates their features in the database.
    Uses file_path as UNIQUE key to avoid duplicates.
    Skips unreadable or unsupported image files.
    Also saves a pickle file with flattened metric data for debugging.
    """
    db = Database(db_path)

    # Create table with UNIQUE file_path
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

    loader = ImageLoader(db, base_dir)
    color_similarity = ColorSimilarity(loader)
    embedding_similarity = EmbeddingSimilarity(loader)
    hashing_similarity = HashingSimilarity(loader)

    metric_data = {}
    count = 0
    skipped_files = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith("._") or not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, base_dir)

            if max_images and count >= max_images:
                print(f"[INFO] Stopped after processing {max_images} images.")
                break

            # Load image (safe)
            try:
                image = loader.load_image_by_path(full_path)
                if image is None:
                    image = Image.open(full_path)
                image = image.convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                print(f"[WARN] Cannot open image {full_path}: {e}")
                skipped_files.append(relative_path)
                continue
            except Exception as e:
                print(f"[ERROR] Unexpected error for {full_path}: {e}")
                skipped_files.append(relative_path)
                continue

            # Compute features
            try:
                color_feature = color_similarity.compute_feature(image)
                embedding_vector = embedding_similarity.compute_feature(image)
                hash_value = hashing_similarity.compute_feature(image)
            except Exception as e:
                print(f"[ERROR] Feature extraction failed for {full_path}: {e}")
                skipped_files.append(relative_path)
                continue

            # Serialize for DB
            color_blob = pickle.dumps(color_feature)
            embedding_blob = pickle.dumps(embedding_vector)
            hash_blob = pickle.dumps(hash_value)

            width, height = image.size
            file_size = os.path.getsize(full_path) if os.path.exists(full_path) else None

            # Insert or update DB
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

            # Flatten features for pickle export
            try:
                (r, g, b), (h, s, l) = color_feature
                color_histogram_flat = np.hstack(
                    [r.flatten(), g.flatten(), b.flatten(),
                     h.flatten(), s.flatten(), l.flatten()]
                ).tolist()
            except Exception:
                color_histogram_flat = None

            metric_data[relative_path] = {
                "color_histogram_flat": color_histogram_flat,
                "embedding": embedding_vector.flatten().tolist()
                if hasattr(embedding_vector, "flatten") else None,
                "hash_hex": str(hash_value) if hash_value is not None else None,
                "width": width,
                "height": height,
                "file_size": file_size,
            }

            count += 1
            print(f"[OK] Processed: {relative_path}")

    db.connection.commit()
    db.close()

    # Save pickle with flattened metrics
    with open("image_metrics.pkl", "wb") as f:
        pickle.dump(metric_data, f)

    # Summary
    print(f"[DONE] Database and pickle updated with {count} images.")
    if skipped_files:
        print(f"[WARN] Skipped {len(skipped_files)} files (unreadable or errors).")
        for sf in skipped_files:
            print(f"   - {sf}")



if __name__ == "__main__":
    base_dir = "/Volumes/BigData03/data"
    scan_and_fill_database(base_dir)
