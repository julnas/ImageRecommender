import os
import pickle
from image_recommender.database import Database
from SimilarityMetrics.color_similarity_metric import ColorSimilarity
from image_recommender.image_loader import ImageLoader

def scan_and_fill_database(base_dir: str, db_path: str = "images.db"):
    """
    This file is used to scan the harddrive for images and fill the database with their paths and ID 
    it creates images.db
    u can use this one time at the gebinning
    """
    db = Database(db_path)

    #create table with ID and file path
    db.cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            color_histogram BLOB
        );
    """)
    db.connection.commit()

    image_id = 1   #starting ID for images
    
    #initialize image loader and color similarity
    loader = ImageLoader(db, base_dir)
    color_similarity = ColorSimilarity(loader)

    max_images = 500 #temporary so we dont load all 500K images (would take to long) 

    #scan the base_dir for image files
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")) and not file.startswith("._"):
                
                if image_id > max_images: #temporary
                    print(f"Stopped after {max_images} images.")
                    break  
                
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_dir)
                
                #load image
                image = loader.load_image_by_path(full_path)
                
                #compute histogram
                histogram = color_similarity.compute_feature(image)
                
                #serialize histogram
                hist_blob = pickle.dumps(histogram)
                
                #import the image into the database
                db.cursor.execute("""
                    INSERT INTO images (image_id, file_path, color_histogram) VALUES (?, ?, ?);
                """, (image_id, relative_path, hist_blob))
                
                print(f"Image {image_id} done ") #to see how much we already loaded

                image_id += 1    #next id
    
    db.connection.commit()
    db.close()
    print("Done DB scan and fill hehe :)")

if __name__ == "__main__":
    base_dir = "/Volumes/BigData03/data"
    scan_and_fill_database(base_dir)
