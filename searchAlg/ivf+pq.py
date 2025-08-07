import sqlite3
import numpy as np
import faiss

def build_ivfpq_index(data, nlist=50, m=8, nbits=8):
    dim = data.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    index.train(data)
    index.add(data)
    index.nprobe = 5
    return index

def search_similar_images_rgb_hsl(db_path, query_rgb, query_hsl, k=5, weight_rgb=0.5, weight_hsl=0.5):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, rgb_histogram, hsl_histogram FROM images")
    rows = cursor.fetchall()
    conn.close()

    ids = []
    rgb_histograms = []
    hsl_histograms = []

    for row in rows:
        ids.append(row[0])
        rgb_histograms.append(np.frombuffer(row[1], dtype=np.float32))
        hsl_histograms.append(np.frombuffer(row[2], dtype=np.float32))

    rgb_histograms = np.array(rgb_histograms).astype('float32')
    hsl_histograms = np.array(hsl_histograms).astype('float32')

    # Zwei Indizes bauen
    index_rgb = build_ivfpq_index(rgb_histograms)
    index_hsl = build_ivfpq_index(hsl_histograms)

    # Suche in RGB
    dist_rgb, idx_rgb = index_rgb.search(query_rgb.astype('float32'), k)
    # Suche in HSL
    dist_hsl, idx_hsl = index_hsl.search(query_hsl.astype('float32'), k)

    # Ergebnisse kombinieren (hier einfach: gewichtetes Mittel der Distanzen)
    combined_scores = {}
    for rank, idx in enumerate(idx_rgb[0]):
        combined_scores[ids[idx]] = combined_scores.get(ids[idx], 0) + dist_rgb[0][rank] * weight_rgb
    for rank, idx in enumerate(idx_hsl[0]):
        combined_scores[ids[idx]] = combined_scores.get(ids[idx], 0) + dist_hsl[0][rank] * weight_hsl

    # Sortieren nach Score
    combined_sorted = sorted(combined_scores.items(), key=lambda x: x[1])
    top_results = combined_sorted[:k]

    return top_results

# --- Beispielaufruf ---
if __name__ == "__main__":
    query_rgb = np.random.rand(1, 64).astype('float32')
    query_hsl = np.random.rand(1, 64).astype('float32')

    results = search_similar_images_rgb_hsl("bilder.db", query_rgb, query_hsl)
    print("Kombinierte Top-Bilder:", results)