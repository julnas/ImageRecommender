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
        """


    def image_generator(self):
        """
        Generator that yields for each valid image.
        """
