from PIL import Image
import numpy as np
import imagehash
import pickle

class HashingSimilarity:
    def __init__(self, loader, hash_size=8):
        self.loader = loader
        self.hash_size = hash_size
        self.bits = hash_size * hash_size

    def compute_feature(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError("Unsupported type")
        return imagehash.average_hash(image, hash_size=self.hash_size)

    def compute_similarity(self, hash1, hash2):
        return 1 - (hash1 - hash2) / float(self.bits)

    def find_similar(self, query_hash, best_k=5):
        similarities = []

        self.loader.db.cursor.execute(
            "SELECT image_id, image_hash FROM images WHERE image_hash IS NOT NULL;"
        )
        rows = self.loader.db.cursor.fetchall()

        for idx, (image_id, hash_blob) in enumerate(rows):
            db_hash = pickle.loads(hash_blob)
            sim = self.compute_similarity(query_hash, db_hash)
            similarities.append((image_id, sim))

            if idx % 100 == 0:
                print(f"Verglichen: {idx} Bilder")

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [img_id for img_id, _ in similarities[:best_k]]
