import sqlite3
import pickle
import os
from federpy.federpy import FederPy

DB_PATH = "images_database.db"
INDEX_DIR = "indexes"

# hier trägst du den Ordner ein, in dem deine Bilder liegen
BASE_DIR = "/Volumes/BigData03/data"

def load_media_urls_from_db(ids_file, db_path, base_dir):
    # IDs-Reihenfolge laden
    with open(ids_file, "rb") as f:
        faiss_ids_in_order = pickle.load(f).tolist()

    # image_id -> file_path aus DB holen
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT image_id, file_path FROM images WHERE file_path IS NOT NULL")
    id2path = dict(cur.fetchall())
    conn.close()

    # Absoluten Pfad aus BASE_DIR + file_path bauen
    media_urls = []
    for i in faiss_ids_in_order:
        if i in id2path:
            full_path = os.path.join(base_dir, id2path[i])
            media_urls.append(full_path)

    return media_urls

def run_feder(index_file, ids_file, db_path, base_dir):
    media_urls = load_media_urls_from_db(ids_file, db_path, base_dir)

    params = {
        "mediaType": "img",
        "mediaUrls": media_urls,
        "width": 1000,
        "height": 640,
        "projectMethod": "umap",
    }

    feder = FederPy(index_file, "faiss", **params)
    feder.setSearchParams({"k": 10, "nprobe": 16}).overview()
    return feder

if __name__ == "__main__":
    index_file = os.path.join(INDEX_DIR, "emb_ivfpq.faiss")
    ids_file = os.path.join(INDEX_DIR, "emb_ivfpq.ids.pkl")

    run_feder(index_file, ids_file, DB_PATH, BASE_DIR)
