from image_recommender.database import Database
from image_recommender.image_loader import ImageLoader
from image_recommender.recomender import Recommender
from SimilarityMetrics.color_similarity_metric import ColorSimilarity
from SimilarityMetrics.embeddings_similarity_metric import EmbeddingSimilarity
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

    # initialize similarity metrics
    metrics = {
        "color": ColorSimilarity(loader),
        "embedding": EmbeddingSimilarity(), #they dont exist yet
        # "hash": HashingSimilarity()         #they dont exist yet
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
