from image_recommender.database import Database
from image_recommender.image_loader import ImageLoader
from image_recommender.recomender import Recommender
from SimilarityMetrics.color_similarity_metric import ColorSimilarity
from SimilarityMetrics.embeddings_similarity_metric import EmbeddingSimilarity
from SimilarityMetrics.hashing_similarity_metric import HashingSimilarity
from PIL import Image
import os


def main():
    # SQLite-Database
    db_path = "/Users/belizsenol/Documents/Uni/UNI Semester 4/BigData/ImageRecommender/images.db"

    # harddrive
    base_dir = "/Volumes/BigData03/data"

    # image to compare
    comparing_image_path = "/Users/belizsenol/Desktop/Unknown.jpeg"
    input_image = Image.open(comparing_image_path).convert("RGB")

    # db connectiom
    db = Database(db_path)

    # ImageLoader
    loader = ImageLoader(db, base_dir)


    # Embeddings (IVFPQ)
    emb = EmbeddingSimilarity(loader)
    # First time (build):
    # emb.build_ivfpq_index("indexes/emb_ivfpq.faiss", nlist=4096, m=16)
    # Normal operation (load):
    # emb.load_ivfpq_index("indexes/emb_ivfpq.faiss")

    # Color (HNSW)
    color = ColorSimilarity(loader, bins=16)
    # First time (build):
    # color.build_hnsw_index("indexes/color_hnsw.bin", m=16, ef_construction=200, ef=100)
    # Normal operation (load):
    # color.load_hnsw_index("indexes/color_hnsw.bin", ef=100)

    # Hash
    hashing = HashingSimilarity(loader)


    # initialize similarity metrics
    metrics = {
    "embedding": emb,
    "color": color,
    "hash": hashing,
    }   

    # recommender system
    recommender = Recommender(db, loader, metrics)

    # how many similar images we want
    best_k = 5

    # get recommendations
    results = recommender.recommend(input_image, best_k=best_k)

    # results
    for metric_name, image_ids in results.items():
        print(f"\nTop {best_k} matches for {metric_name}:")
        for idx, img_id in enumerate(image_ids, start=1):
            print(f"{idx}. Image ID: {img_id}")
            # load the images
            img = loader.load_image(img_id)
            img.show(title=f"{metric_name} Match #{idx}")

    """
    #testing if the image will be leaded 
    test_id = 3
    image = loader.load_image(test_id)

    if image:
        print(f"Loaded image with ID {test_id} yaayy")
        image.show() 
    else:
        print(f"Nooo it failed to load image with ID {test_id}")
    """

    # closing connection to db
    db.close()


if __name__ == "__main__":
    main()
