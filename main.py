import os
# Fix for MacOS OpenMP duplicate lib issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from image_recommender.database import Database
from image_recommender.image_loader import ImageLoader
from image_recommender.recomender import Recommender
from SimilarityMetrics.color_similarity_metric import ColorSimilarity
from SimilarityMetrics.embeddings_similarity_metric import EmbeddingSimilarity
from SimilarityMetrics.hashing_similarity_metric import HashingSimilarity
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def evaluate_ann_recall(recommender, ann_results, k=5):
    """Compare ANN top-k results with exact top-k database searches."""
    for metric_name in ("color", "embedding"):
        metric = recommender.metrics.get(metric_name)
        query_feature = recommender.last_query_features.get(metric_name)

        if metric is None or query_feature is None:
            continue

        exact_search_start = perf_counter()
        exact_ids = metric.find_similar_exact(query_feature, best_k=k)
        exact_search_seconds = perf_counter() - exact_search_start

        ann_ids = ann_results.get(metric_name, [])[:k]
        denominator = len(exact_ids)
        overlap = len(set(ann_ids) & set(exact_ids))
        recall = overlap / denominator if denominator else 0.0

        print(
            f"[QUALITY] {metric_name}: Recall@{k}={recall:.4f} "
            f"({overlap}/{denominator}), "
            f"exact_search={exact_search_seconds:.4f}s"
        )

def _make_vertical_montage(paths, max_width=320, pad=6, bg=255):
    """
    Baut ein einzelnes PIL-Image als vertikale Montage aus mehreren Pfaden.
    - max_width: jede Referenz wird proportional auf diese Breite skaliert
    - pad: Innenabstand in Pixeln
    - bg: Hintergrund (255 = weiß)
    """
    imgs = []
    for p in paths:
        try:
            im = Image.open(p)
            # Orientierung aus EXIF berücksichtigen
            im = ImageOps.exif_transpose(im).convert("RGB")
            # proportional auf max_width skalieren
            w, h = im.size
            if w != max_width:
                new_h = max(1, int(h * (max_width / float(w))))
                im = im.resize((max_width, new_h), Image.BILINEAR)
            imgs.append(im)
        except Exception:
            # Platzhalter bei Ladefehler
            ph = Image.new("RGB", (max_width, 60), (255, 200, 200))
            imgs.append(ph)

    if not imgs:
        return Image.new("RGB", (max_width, 60), (230, 230, 230))

    total_h = sum(im.size[1] for im in imgs) + pad * (len(imgs) + 1)
    canvas = Image.new("RGB", (max_width + 2*pad, total_h), (bg, bg, bg))

    y = pad
    for im in imgs:
        canvas.paste(im, (pad, y))
        y += im.size[1] + pad

    return canvas

def plot_topk_basic(results, best_k, loader, comparing_image_path):
    """
    Eine Zeile pro Metrik, eine Referenzspalte (Montage aus 1..N Referenzen),
    daneben best_k Ergebnis-Spalten. ALLES in EINEM Figure-Fenster.
    """
    metrics = list(results.keys())
    n_rows = len(metrics)
    n_cols = best_k + 1  # 1 Referenzspalte + k Treffer

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))

    # Immer 2D-Array sicherstellen
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Referenz-Pfade normalisieren
    if isinstance(comparing_image_path, (list, tuple)):
        ref_paths = list(comparing_image_path)
    else:
        ref_paths = [comparing_image_path]

    # Vorab eine Montage bauen (ein einziges Bild), das wir in jeder Row links anzeigen
    ref_panel = _make_vertical_montage(ref_paths, max_width=320, pad=6)
    ref_panel_np = np.asarray(ref_panel)

    for r, metric_name in enumerate(metrics):
        # 1. Spalte: die Montage
        ax_ref = axes[r, 0]
        ax_ref.imshow(ref_panel_np)
        ax_ref.axis('off')
        ax_ref.set_title(f"{metric_name}\nReferenz(en)", fontsize=9)

        # Restliche Spalten: Top-k Ergebnisse
        image_ids = results[metric_name]
        for c in range(best_k):
            ax = axes[r, c+1]
            ax.axis('off')

            if c < len(image_ids):
                img_id = image_ids[c]
                img = loader.load_image(img_id)  # sollte ein PIL-Image oder np.array liefern
                if img is not None:
                    if isinstance(img, Image.Image):
                        img = ImageOps.exif_transpose(img).convert("RGB")
                        img = np.asarray(img)
                    ax.imshow(img)
                ax.set_title(f"#{c+1} ID: {img_id}", fontsize=9)
            else:
                ax.set_title(f"#{c+1} (leer)", fontsize=9)

    plt.tight_layout()
    plt.show()


def main():
    """Hauptfunktion zum Ausführen der Bildempfehlung."""

    time_start = perf_counter()
    # ------------------------ CONFIG ------------------------
    db_path = "images_database.db"
    base_dir = "/Volumes/BigData03/data" # Basisverzeichnis für Bilder
    comparing_image_path = [ "/Users/belizsenol/Downloads/IMG_8203.JPG"] # Pfad oder Liste von Pfaden zu Vergleichsbildern
    best_k = 5 # Anzahl der besten Ergebnisse, die zurückgegeben werden sollen
    measure_recall = True  # Temporär: Recall@5 gegen exakte Suche messen

    # ------------------------ DB + Loader ------------------------
    db = Database(db_path)
    loader = ImageLoader(db, base_dir)

    # ------------------------ Embedding (IVFPQ) ------------------------
    emb = EmbeddingSimilarity(loader)
    index_load_start = perf_counter()
    emb.load_ivfpq_index("indexes/emb_ivfpq.faiss")
    print(
        f"[PROFILE] embedding index load: "
        f"{perf_counter() - index_load_start:.4f}s"
    )


    # ------------------------ Color (FAISS-HNSW) ------------------------
    color = ColorSimilarity(loader, bins=16)
    index_load_start = perf_counter()
    color.load_faiss_hnsw_index("indexes/color_hnsw.faiss", ef=100)
    print(
        f"[PROFILE] color index load: "
        f"{perf_counter() - index_load_start:.4f}s"
    )


    # ------------------------ Hash ------------------------
    hashing = HashingSimilarity(loader)
    hash_cache_load_start = perf_counter()
    cached_hash_count = hashing.load_cache()
    print(
        f"[PROFILE] hashing cache load: "
        f"{perf_counter() - hash_cache_load_start:.4f}s "
        f"({cached_hash_count} hashes)"
    )

    # ------------------------ Choose metrics ------------------------
    # Einzelne Metriken aktivieren/deaktivieren
    metrics = {"color": color, "embedding": emb, "hashing": hashing}

    # ------------------------ Recommendation ------------------------
    recommender = Recommender(db, loader, metrics)
    if isinstance(comparing_image_path, list):
        input_image = []
        for image in comparing_image_path:
            input_image.append(Image.open(image).convert("RGB"))
    else:
        input_image = Image.open(comparing_image_path).convert("RGB")
    results = recommender.recommend(input_image, best_k=best_k)
    recommendation_end = perf_counter()
    print(f"[INFO] Recommendation took {recommendation_end - time_start:.2f} seconds")

    if measure_recall:
        evaluate_ann_recall(recommender, results, k=5)

    # ------------------------ Output ------------------------
    plot_topk_basic(results, best_k, loader, comparing_image_path)

    db.close()



if __name__ == "__main__":
    main()
