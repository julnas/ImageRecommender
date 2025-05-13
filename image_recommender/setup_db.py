import os
from database import Database

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
            file_path TEXT NOT NULL
        );
    """)
    db.connection.commit()

    image_id = 1   #starting ID for images

    #scan the base_dir for image files
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")) and not file.startswith("._"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_dir)
                #import the image into the database
                db.cursor.execute("""
                    INSERT INTO images (image_id, file_path) VALUES (?, ?)
                """, (image_id, relative_path))

                image_id += 1    #next id
    
    #testing get_all_image_ids()
    ids = db.get_all_image_ids()
    for image_id in ids:
        print(f"→ ID: {image_id}, Path: {db.get_image_path(image_id)}")

    db.connection.commit()
    db.close()
    print("Done DB scan and fill hehe :)")

if __name__ == "__main__":
    base_dir = "/Volumes/BigData03/data"
    scan_and_fill_database(base_dir)
