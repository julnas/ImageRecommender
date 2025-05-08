import os
from PIL import Image
from typing import Generator

class ImageLoader:
    def __init__(self, db, base_dir: str):
        """
        Loads images from disk based on paths stored in the database.

        Parameters:
        db: an instance of DatabaseManager
        base_dir: root path to the external image folder (e.g. '/media/usbdrive/images/')
        """
        self.db = db
        self.base_dir = base_dir

    def load_image(self, image_id):
        """
        Loads an image by ID using the database and base_dir.
        
        Parameters:
        image_id: the ID of the image im the database

        Returns:
        A PIL Image object (if successful, or none) 
        """

        #get relative file path of the image from the database
        relative_path = self.db.get_image_path(image_id)

        #If no path is found
        if not relative_path:
            print(f"No path found for image ID {image_id}")
            return None

        #build the full absolute file path
        full_path = os.path.join(self.base_dir, relative_path)

        #check if the image file actually exists on disk
        if os.path.exists(full_path):
             #load the image using PIL and convert it to RGB
            return Image.open(full_path).convert("RGB")
        else:
            # Print a warning if the file doesn't exist at the given path
            print(f"File does not exist: {full_path}")

        # If anything fails, return None
        return None


    def image_generator(self) -> Generator[tuple[int, Image.Image], None, None]:
        """
        Generator that yields for each valid image.
        
        Yields:
        A tuple of (image_id, image) for each successfully loaded image
        """
        
        #go throught all image IDs stored in the database
        for image_id in self.db.get_all_image_ids():

            #try loading the image corresponding to the current ID
            img = self.load_image(image_id)

            #if image loading was successful, yield it
            if img is not None:
                yield image_id, img  #yields a tuple of image ID and the PIL Image object