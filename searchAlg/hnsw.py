import sqlite3
import numpy as np
import faiss

def search_similar_images(db_path, query_embedding, k=5):
    # --- Verbindung zur übergebenen Datenbank ---
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Embeddings aus DB laden ---
    cursor.execute("SELECT id, embedding FROM images")
    rows = cursor.fetchall()
    conn.close()

    ids = []
    embeddings = []
    for row in rows:
        ids.append(row[0])
        embeddings.append(np.frombuffer(row[1], dtype=np.float32))
    
    embeddings = np.array(embeddings).astype('float32')

    # --- HNSW Index erstellen ---
    dim = embeddings.shape[1]
    m = 32
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 50

    index.add(embeddings)

    # --- Suche ---
    distances, idx_positions = index.search(query_embedding.astype('float32'), k)
    top_ids = [ids[i] for i in idx_positions[0]]

    return top_ids, distances[0]