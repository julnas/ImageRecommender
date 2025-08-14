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


def main():
    # ------------------------ CONFIG ------------------------
    db_path = "/Users/jule/Documents/Uni/4. Semester/Big Data Engineering/ImageRecommender/images.db"
    base_dir = "/Volumes/BigData03/data"
    comparing_image_path = "/Users/jule/Downloads/jule03_bearb.jpg"
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

    # ------------------------ Debug dimension check ------------------------
    if hasattr(emb, "faiss_index"):
        print("[DEBUG] IVFPQ dim:", emb.faiss_index.d)
    if hasattr(color, "faiss_index"):
        print("[DEBUG] HNSW dim:", color.faiss_index.d)

    # ------------------------ Choose metrics ------------------------
    # 👉 Zum Testen einzelne Metriken aktivieren/deaktivieren
    metrics = {"color": color, "embedding": emb, "hashing": hashing}

    # ------------------------ Recommendation ------------------------
    recommender = Recommender(db, loader, metrics)
    input_image = Image.open(comparing_image_path).convert("RGB")
    results = recommender.recommend(input_image, best_k=best_k)

    # ------------------------ Output ------------------------
    for metric_name, image_ids in results.items():
        print(f"\nTop {best_k} matches for {metric_name}:")
        for idx, img_id in enumerate(image_ids, start=1):
            print(f"{idx}. Image ID: {img_id}")
            img = loader.load_image(img_id)
            if img:
                img.show(title=f"{metric_name} Match #{idx}")

    db.close()


if __name__ == "__main__":
    main()
