import os
import pickle
from image_recommender.database import Database
from SimilarityMetrics.color_similarity_metric import ColorSimilarity
#from SimilarityMetrics.embeddings_similarity_metric import embedding_similarity
#from SimilarityMetrics.hashing_similarity_metric import HashingSimilarity
from image_recommender.image_loader import ImageLoader


def scan_and_fill_database(base_dir: str, db_path: str = "images.db"):
    """
    This file is used to scan the harddrive for images and fill the database with their paths and ID
    it creates images.db
    u can use this one time at the gebinning
    """
    db = Database(db_path)
    metric_data = {}  # image_id → dict of metrics

    # create table with ID and file path
    db.cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            color_histogram BLOB,
            width INTEGER,
            height INTEGER,
            file_size INTEGER
        );
    """
    )
    db.connection.commit()

    image_id = 1  # starting ID for images

    # initialize image loader and color similarity
    loader = ImageLoader(db, base_dir)
    color_similarity = ColorSimilarity(loader)

    max_images = (
        2000  # temporary so we dont load all 500K images (would take to long)
    )

    # scan the base_dir for image files
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")) and not file.startswith(
                "._"
            ):

                if image_id > max_images:  # temporary
                    print(f"Stopped after {max_images} images.")
                    break

                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_dir)

                # load image
                image = loader.load_image_by_path(full_path)

                # compute features
                color_histogram = color_similarity.compute_feature(image)
                #embedding_vector = embedding_similarity.compute_embedding(image)
                #hash_vector = HashingSimilarity.compute_hash(image)

                # serialize for DB
                color_blob = pickle.dumps(color_histogram)
                #embedding_blob = pickle.dumps(embedding_vector)
                #hash_blob = pickle.dumps(hash_vector)

                # get resolution
                width, height = image.size if image else (None, None)

                # get file size
                file_size = (
                    os.path.getsize(full_path) if os.path.exists(full_path) else None
                )

                # import the image into the database
                db.cursor.execute(
                    """
                    INSERT INTO images (image_id, file_path, color_histogram, width, height, file_size) 
                    VALUES (?, ?, ?, ?, ?, ?);
                """,
                    (image_id, relative_path, color_blob, width, height, file_size),
                )
                
                # for the pickle file
                metric_data[image_id] = {
                    "color_histogram": color_histogram.flattern().tolist() if hasattr(color_histogram, 'tolist') else color_histogram #,
                    #"embedding": embedding_vector.flattern().tolist() if hasattr(embedding_vector, 'tolist') else embedding_vector,
                    #"hash": hash_vector.flattern().tolist() if hasattr(hash_vector, 'tolist') else hash_vector
                    }

                print(f"Image {image_id} done ")  # to see how much we already loaded

                image_id += 1  # next id

    db.connection.commit()
    db.close()
    
    with open("image_metrics.pkl", "wb") as f:
        pickle.dump(metric_data, f)
    
    print(" DB and pickle file done hehe :)")


if __name__ == "__main__":
    base_dir = "/Volumes/BigData03/data"
    scan_and_fill_database(base_dir)
