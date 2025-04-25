from image_recommender.database import Database
from image_recommender.image_loader import ImageLoader
import os

def main():
    #SQLite-Database
    db_path = "/Users/belizsenol/Documents/Uni/UNI Semester 4/BigData/ImageRecommender/images.db"

    #harddrive
    base_dir = "/Volumes/BigData03/data/DIV2k/DIV2K_train_HR"  

    #db connectiom
    db = Database(db_path)

    #ImageLoader
    loader = ImageLoader(db, base_dir)
    

    #testing if the image will be leaded 
    test_id = 3
    image = loader.load_image(test_id)

    if image:
        print(f"Loaded image with ID {test_id} yaayy")
        image.show()  # Zeigt das Bild in einem Viewer-Fenster
    else:
        print(f"Nooo it failed to load image with ID {test_id}")

    # Datenbank schließen
    db.close()

if __name__ == "__main__":
    main()
