# implement the hashing similarity metric
#imports
from PIL import Image
import imagehash

class HashingSimilarityMetric:
    def __init__(self, hash_size=8):
        self.hash_size = hash_size

    def create_hash(self, image_path):
        """
        Create a perceptual hash for the image at the given path.
        :param image_path: Path to the image file.
        :return: Perceptual hash of the image.
        """
        image = Image.open(image_path)
        return imagehash.average_hash(image, hash_size=self.hash_size)
    
    def compute_similarity(self, hash1, hash2):
        """
        Compute the similarity between two perceptual hashes.
        :param hash1: First perceptual hash.
        :param hash2: Second perceptual hash.
        :return: Similarity score (0 to 1).
        """
        return 1 - (hash1 - hash2) / len(hash1.hash) ** 2


"""def main():
    # Replace these paths with your actual image file paths
    image_path1 = "/Users/jule/Downloads/PHOTO-2024-07-23-15-12-40.jpg"
    image_path2 = "/Users/jule/Downloads/uvex-Schutzbrillen-Gefahren-von-blauem-UV-Licht.jpg"

    metric = HashingSimilarityMetric()

    hash1 = metric.create_hash(image_path1)
    hash2 = metric.create_hash(image_path2)

    print(f"Hash 1: {hash1}")
    print(f"Hash 2: {hash2}")

    similarity = metric.compute_similarity(hash1, hash2)
    print(f"Similarity score: {similarity}")

if __name__ == "__main__":
    main()"""