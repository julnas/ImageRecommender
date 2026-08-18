import os
import pickle
import sqlite3
import tempfile
from typing import Optional

from PIL import Image, UnidentifiedImageError, ImageFile

# Allow loading partially corrupted/truncated image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True

from image_recommender.database import Database
from image_recommender.image_loader import ImageLoader
from SimilarityMetrics.color_similarity_metric import ColorSimilarity
from SimilarityMetrics.embeddings_similarity_metric import EmbeddingSimilarity
from SimilarityMetrics.hashing_similarity_metric import HashingSimilarity


# ----------------- Supported image formats -----------------

ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}
# Extend if needed:
# ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic"}


# ----------------- Image path generator -----------------

def iter_image_paths(
    base_dir: str,
    follow_links: bool = False,
    exclude: Optional[list[str]] = None,
):
    """
    Yield absolute paths to image files below base_dir recursively.

    exclude supports:
      - absolute directory paths
      - relative paths from base_dir
      - directory names
    """

    exclude = exclude or []

    abs_exclude_paths = set()
    rel_exclude_paths = set()
    exclude_names = set()

    base_dir_abs = os.path.normpath(
        os.path.abspath(base_dir)
    )

    for item in exclude:
        if os.path.isabs(item):
            path = os.path.normpath(item)

            abs_exclude_paths.add(path)
            exclude_names.add(
                os.path.basename(path)
            )

        else:
            relative_path = os.path.normpath(item)

            rel_exclude_paths.add(
                relative_path
            )

            abs_exclude_paths.add(
                os.path.normpath(
                    os.path.join(
                        base_dir_abs,
                        relative_path,
                    )
                )
            )

            exclude_names.add(
                os.path.basename(relative_path)
            )

    for root, dirs, files in os.walk(
        base_dir_abs,
        followlinks=follow_links,
    ):
        root_abs = os.path.normpath(root)

        # Filter directories in-place so excluded directories
        # are never traversed.
        kept_dirs = []

        for directory in dirs:
            child_abs = os.path.normpath(
                os.path.join(
                    root_abs,
                    directory,
                )
            )

            child_rel = os.path.normpath(
                os.path.relpath(
                    child_abs,
                    base_dir_abs,
                )
            )

            if (
                child_abs in abs_exclude_paths
                or child_rel in rel_exclude_paths
                or directory in exclude_names
            ):
                continue

            kept_dirs.append(directory)

        dirs[:] = kept_dirs

        for filename in files:
            if filename.startswith("._"):
                continue

            if (
                os.path.splitext(filename)[1].lower()
                in ALLOWED_EXTS
            ):
                yield os.path.join(
                    root_abs,
                    filename,
                )


# ----------------- Database schema -----------------

def create_database_schema(db: Database):
    """
    Create the database schema in the given database.
    """

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


# ----------------- Database setup -----------------

def scan_and_fill_database(
    base_dir: str,
    db_path: str = "images_database.db",
    max_images: Optional[int] = None,
    commit_batch_size: int = 5000,
):
    """
    Build a new temporary database from the selected drive.

    The existing database remains untouched until the complete
    scan has successfully finished.

    Images that are no longer present on the selected drive
    will therefore not exist in the new database.

    Features are calculated in a streaming fashion and
    transactions are committed in batches to keep memory usage
    manageable for very large image collections.

    Returns:
        True  -> setup succeeded
        False -> setup failed
    """

    base_dir = os.path.normpath(
        os.path.abspath(base_dir)
    )

    if not os.path.isdir(base_dir):
        print(
            f"[ERROR] Base directory does not exist: "
            f"{base_dir}"
        )
        return False

    if commit_batch_size <= 0:
        raise ValueError(
            "commit_batch_size must be greater than 0."
        )

    db_path = os.path.abspath(db_path)

    db_directory = os.path.dirname(db_path)

    if not db_directory:
        db_directory = os.getcwd()

    os.makedirs(
        db_directory,
        exist_ok=True,
    )

    temp_db_path = None
    temp_db = None

    count = 0
    skipped = 0
    batch_count = 0

    try:
        # -----------------------------------------------------
        # Create the temporary database in the same directory.
        #
        # This allows the final replacement to be atomic.
        # -----------------------------------------------------

        fd, temp_db_path = tempfile.mkstemp(
            prefix=".images_database_",
            suffix=".tmp.db",
            dir=db_directory,
        )

        os.close(fd)

        print(
            f"[INFO] Creating temporary database: "
            f"{temp_db_path}"
        )

        # -----------------------------------------------------
        # Open temporary database
        # -----------------------------------------------------

        temp_db = Database(
            temp_db_path
        )

        temp_db.cursor.execute(
            "PRAGMA journal_mode=WAL;"
        )

        temp_db.cursor.execute(
            "PRAGMA synchronous=NORMAL;"
        )

        temp_db.cursor.execute(
            "PRAGMA busy_timeout=10000;"
        )

        create_database_schema(
            temp_db
        )

        # -----------------------------------------------------
        # Initialize image loader and feature extractors
        # -----------------------------------------------------

        loader = ImageLoader(
            temp_db,
            base_dir,
        )

        color_similarity = ColorSimilarity(
            loader
        )

        embedding_similarity = EmbeddingSimilarity(
            loader
        )

        hashing_similarity = HashingSimilarity(
            loader
        )

        # -----------------------------------------------------
        # Scan the selected drive
        # -----------------------------------------------------

        for full_path in iter_image_paths(
            base_dir=base_dir,
            follow_links=False,
            exclude=[],
        ):

            if (
                max_images is not None
                and count >= max_images
            ):
                print(
                    f"[INFO] Stopped after "
                    f"{max_images} images."
                )
                break

            relative_path = os.path.normpath(
                os.path.relpath(
                    full_path,
                    base_dir,
                )
            )

            # -------------------------------------------------
            # Load image
            # -------------------------------------------------

            try:
                img = loader.load_image_by_path(
                    full_path
                )

                if img is None:
                    with Image.open(full_path) as im:
                        img = im.convert(
                            "RGB"
                        ).copy()

                else:
                    img = img.convert(
                        "RGB"
                    ).copy()

            except (
                UnidentifiedImageError,
                OSError,
            ) as e:

                print(
                    f"[WARN] Skip (cannot open): "
                    f"{relative_path} -> {e}"
                )

                skipped += 1
                continue

            except Exception as e:

                print(
                    f"[WARN] Skip (open exception): "
                    f"{relative_path} -> {e}"
                )

                skipped += 1
                continue

            # -------------------------------------------------
            # Calculate features
            # -------------------------------------------------

            try:
                color_feature = (
                    color_similarity.compute_feature(
                        img
                    )
                )

                embedding_vector = (
                    embedding_similarity.compute_feature(
                        img
                    )
                )

                hash_value = (
                    hashing_similarity.compute_feature(
                        img
                    )
                )

            except Exception as e:

                print(
                    f"[WARN] Skip (feature error): "
                    f"{relative_path} -> {e}"
                )

                skipped += 1
                continue

            # -------------------------------------------------
            # Serialize features for database storage
            #
            # Color and embedding features still use Pickle.
            #
            # The perceptual hash is different:
            # it is stored directly as an 8-byte BLOB.
            # -------------------------------------------------

            try:
                color_blob = pickle.dumps(
                    color_feature
                )

                embedding_blob = pickle.dumps(
                    embedding_vector
                )

                hash_blob = (
                    hashing_similarity.hash_to_blob(
                        hash_value
                    )
                )

            except Exception as e:

                print(
                    f"[WARN] Skip (serialization error): "
                    f"{relative_path} -> {e}"
                )

                skipped += 1
                continue

            # -------------------------------------------------
            # Read image metadata
            # -------------------------------------------------

            width, height = img.size

            try:
                file_size = os.path.getsize(
                    full_path
                )

            except OSError:
                file_size = None

            # -------------------------------------------------
            # Insert image into temporary database
            # -------------------------------------------------

            try:
                temp_db.cursor.execute(
                    """
                    INSERT INTO images (
                        file_path,
                        color_histogram,
                        embedding,
                        image_hash,
                        width,
                        height,
                        file_size
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)

                    ON CONFLICT(file_path) DO UPDATE SET
                        color_histogram =
                            excluded.color_histogram,
                        embedding =
                            excluded.embedding,
                        image_hash =
                            excluded.image_hash,
                        width =
                            excluded.width,
                        height =
                            excluded.height,
                        file_size =
                            excluded.file_size;
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

                print(
                    f"[WARN] DB write failed for "
                    f"{relative_path}: {e}"
                )

                skipped += 1
                continue

            count += 1
            batch_count += 1

            # -------------------------------------------------
            # Commit periodically.
            #
            # This is important for very large collections
            # because the database may contain 500,000+
            # images.
            # -------------------------------------------------

            if batch_count >= commit_batch_size:

                temp_db.connection.commit()

                print(
                    f"[OK] Committed batch of "
                    f"{batch_count} "
                    f"(total processed: {count}, "
                    f"skipped: {skipped})"
                )

                batch_count = 0

            # -------------------------------------------------
            # Print progress information.
            # -------------------------------------------------

            if count % 200 == 0:

                print(
                    f"[INFO] Processed so far: "
                    f"{count} "
                    f"(skipped: {skipped})"
                )

        # -----------------------------------------------------
        # Commit remaining images.
        # -----------------------------------------------------

        if batch_count > 0:

            temp_db.connection.commit()

            print(
                f"[OK] Final commit of "
                f"{batch_count} "
                f"(total processed: {count}, "
                f"skipped: {skipped})"
            )

        # -----------------------------------------------------
        # Do not replace the production database when only a
        # partial scan was requested.
        # -----------------------------------------------------

        if (
            max_images is not None
            and count >= max_images
        ):

            print(
                "[WARNING] Partial scan detected. "
                "Temporary database will NOT replace "
                "the existing database."
            )

            return False

        # -----------------------------------------------------
        # Full scan completed successfully.
        # -----------------------------------------------------

        print(
            f"[DONE] Full scan completed. "
            f"Processed: {count}, "
            f"skipped: {skipped}."
        )

        # -----------------------------------------------------
        # Close temporary database before replacing it.
        # -----------------------------------------------------

        temp_db.close()
        temp_db = None

        # -----------------------------------------------------
        # Replace production database.
        # -----------------------------------------------------

        print(
            "[INFO] Replacing existing database "
            "with the new database..."
        )

        os.replace(
            temp_db_path,
            db_path,
        )

        temp_db_path = None

        print(
            "[DONE] Database setup completed "
            "successfully."
        )

        print(
            "[INFO] Hashes are stored using the "
            "compact 8-byte representation."
        )

        return True

    except Exception as e:

        print(
            f"[ERROR] Database setup failed: {e}"
        )

        print(
            "[INFO] Existing database was preserved."
        )

        return False

    finally:

        # -----------------------------------------------------
        # Close temporary database if still open.
        # -----------------------------------------------------

        if temp_db is not None:

            try:
                temp_db.close()

            except Exception as e:

                print(
                    f"[WARN] Failed to close temporary "
                    f"database: {e}"
                )

        # -----------------------------------------------------
        # Remove incomplete temporary database.
        # -----------------------------------------------------

        if (
            temp_db_path is not None
            and os.path.exists(temp_db_path)
        ):

            try:
                os.remove(
                    temp_db_path
                )

                print(
                    "[INFO] Temporary database removed."
                )

            except OSError as e:

                print(
                    f"[WARN] Could not remove temporary "
                    f"database {temp_db_path}: {e}"
                )


# ----------------- Example -----------------

if __name__ == "__main__":

    base_dir = "/Volumes/BigData03/data"

    success = scan_and_fill_database(
        base_dir=base_dir,
        db_path="images_database.db",
        max_images=None,
        commit_batch_size=5000,
    )

    if success:

        print(
            "[OK] Database setup finished successfully."
        )

        print(
            "[INFO] Next step: rebuild FAISS indexes."
        )

    else:

        print(
            "[ERROR] Database setup failed."
        )
