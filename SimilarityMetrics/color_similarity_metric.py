# imports
import numpy as np
import cv2
from scipy.spatial.distance import cosine
import pickle


class ColorSimilarity:
    def __init__(self, loader, bins=16):
        self.loader = loader
        self.bins = bins

    # --------------------------------------------------
    # Function: calculate_histogram
    # Description:
    # This function computes color histograms for each channel (Red, Green, Blue)
    # of the given image. Each histogram is divided into a specified number of bins
    # (default: 16). After computing, the histograms are normalized so that their sum equals 1.
    # This ensures the histograms are comparable across different image sizes.
    # --------------------------------------------------
    def calculate_histogram(self, image, bins=16):
        # Calculate histograms for Red, Green, and Blue channels
        hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])

        # Normalize histograms so that the sum equals 1
        hist_r /= hist_r.sum()
        hist_g /= hist_g.sum()
        hist_b /= hist_b.sum()

        return hist_r, hist_g, hist_b

    # --------------------------------------------------
    # Function: calculate_similarity
    # Description:
    # This function calculates the similarity between two histograms using cosine similarity.
    # Cosine similarity measures the cosine of the angle between two non-zero vectors.
    # It returns a value between -1 (completely different) and 1 (identical).
    # --------------------------------------------------
    def calculate_similarity(self, hist1, hist2):
        similarity = 1 - cosine(hist1.flatten(), hist2.flatten())
        return similarity

    # --------------------------------------------------
    # Function: compute_feature
    # Description:
    # converts a given input image (originally in PIL format) into a array
    # It then extracts color histograms for each channel
    # ---------------------------------------------------
    def compute_feature(self, image):
        if image is None:
            return None

        # convert PIL → NumPy array (RGB)
        image_np = np.array(image)

        # it has to be uint8
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)

        # convert RGB to BGR for OpenCV compatibility
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self.calculate_histogram(image_bgr, bins=self.bins)

    # --------------------------------------------------
    # Function: find_similar
    # Description:
    # This function calculates the overall color similarity and IDs for every Image.
    # It first computes the color histograms for each image (Red, Green, Blue),
    # then measures the similarity per channel using cosine similarity.
    # Finally, it combines these similarities using weighted contributions (as used in luminance calculations):
    # Red (30%), Green (59%), Blue (11%).
    # it appendds the id and the similarity to a list and sorts it.
    # The top-k most similar images are returned.
    # --------------------------------------------------
    def find_similar(self, query_hist, best_k=5):
        similarities = []

        # get all images with color histograms from the database
        self.loader.db.cursor.execute(
            "SELECT image_id, color_histogram FROM images WHERE color_histogram IS NOT NULL;"
        )
        rows = self.loader.db.cursor.fetchall()

        for idx, (image_id, hist_blob) in enumerate(rows):
            db_hist = pickle.loads(hist_blob)  # unpickle the histogram

            # Calculate cosine similarity for each color channel
            red_similarity = self.calculate_similarity(query_hist[0], db_hist[0])
            green_similarity = self.calculate_similarity(query_hist[1], db_hist[1])
            blue_similarity = self.calculate_similarity(query_hist[2], db_hist[2])

            # Combine the similarities using weighted average (standard RGB luminance weights)
            compl_similarity = (
                0.3 * red_similarity + 0.59 * green_similarity + 0.11 * blue_similarity
            )

            # append every image id and the similarity to a list, so we can compsare them later
            similarities.append((image_id, compl_similarity))

            # counting the pictures
            if idx % 100 == 0:
                print(f"Compared: {idx} piktures")

        # Sort descending by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [img_id for img_id, _ in similarities[:best_k]]


"""
# --------------------------------------------------
# Load two images from file paths
# --------------------------------------------------
filename1 = "/Users/jule/Downloads/Elbe_-_flussaufwärts_kurz_nach_Ort_Königstein.jpg"
filename2 = '/Users/jule/Downloads/red-background.png'

# Read the images using OpenCV (in BGR format)
image1 = cv2.imread(filename1)
image2 = cv2.imread(filename2)

# Convert BGR (OpenCV default) to RGB
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Compute and print the overall similarity
similarity = complete_similarity(image1, image2, bins=16)
print("The similarity between image 1 and 2 is:", similarity)
"""
