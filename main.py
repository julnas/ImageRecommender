import os
# Fix for MacOS OpenMP duplicate lib issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from image_recommender.database import Database
from image_recommender.image_loader import ImageLoader
from image_recommender.recomender import Recommender
from SimilarityMetrics.color_similarity_metric import ColorSimilarity
from SimilarityMetrics.embeddings_similarity_metric import EmbeddingSimilarity
from SimilarityMetrics.hashing_similarity_metric import HashingSimilarity
from PIL import Image

import matplotlib.pyplot as plt

def plot_topk_basic(results, best_k, loader, comparing_image_path):
    metrics = list(results.keys())
    n_rows = len(metrics)
    n_cols = best_k + 1  # +1 für das Vergleichsbild

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))

    # Falls nur eine Zeile/Spalte, immer 2D-Liste erzeugen
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Vergleichsbild laden
    ref_img = Image.open(comparing_image_path)

    for r, metric_name in enumerate(metrics):
        # 1. Spalte: Vergleichsbild
        ax_ref = axes[r][0]
        ax_ref.axis('off')
        ax_ref.imshow(ref_img)
        ax_ref.set_title(f"{metric_name}\nReferenz", fontsize=9)

        # Rest: Top-k Bilder
        image_ids = results[metric_name]
        for c in range(best_k):
            ax = axes[r][c+1]  # +1 wegen Referenzspalte
            ax.axis('off')

            if c < len(image_ids):
                img_id = image_ids[c]
                img = loader.load_image(img_id)
                if img is not None:
                    ax.imshow(img)
                ax.set_title(f"#{c+1} ID: {img_id}", fontsize=9)
            else:
                ax.set_title(f"#{c+1} (leer)", fontsize=9)

    plt.tight_layout()
    plt.show()





def main():
    # ------------------------ CONFIG ------------------------
    db_path = "/Users/jule/Documents/Uni/4. Semester/Big Data Engineering/ImageRecommender/images_database.db"
    base_dir = "/Volumes/BigData03/data"
    comparing_image_path = "/Users/jule/Downloads/PHOTO-2025-05-24-17-06-36.jpg"
    best_k = 5

    # ------------------------ DB + Loader ------------------------
    db = Database(db_path)
    loader = ImageLoader(db, base_dir)

    # ------------------------ Embedding (IVFPQ) ------------------------
    emb = EmbeddingSimilarity(loader)
    emb.load_ivfpq_index("indexes/emb_ivfpq.faiss")


    # ------------------------ Color (FAISS-HNSW) ------------------------
    color = ColorSimilarity(loader, bins=16)
    color.load_faiss_hnsw_index("indexes/color_hnsw.faiss", ef=100)


    # ------------------------ Hash ------------------------
    hashing = HashingSimilarity(loader)

    # ------------------------ Choose metrics ------------------------
    # 👉 Zum Testen einzelne Metriken aktivieren/deaktivieren
    metrics = {"color": color, "embedding": emb, "hashing": hashing}

    # ------------------------ Recommendation ------------------------
    recommender = Recommender(db, loader, metrics)
    input_image = Image.open(comparing_image_path).convert("RGB")
    results = recommender.recommend(input_image, best_k=best_k)

    # ------------------------ Output ------------------------
    plot_topk_basic(results, best_k, loader, comparing_image_path)

    db.close()



if __name__ == "__main__":
    main()
