import numpy as np
import cv2
from scipy.spatial.distance import cosine
import pickle
from PIL import Image

class ColorSimilarity:
    def __init__(self, loader, bins=16):
        self.loader = loader
        self.bins = bins

    def calculate_rgb_histogram(self, image_bgr):
        hist_r = cv2.calcHist([image_bgr], [2], None, [self.bins], [0, 256])
        hist_g = cv2.calcHist([image_bgr], [1], None, [self.bins], [0, 256])
        hist_b = cv2.calcHist([image_bgr], [0], None, [self.bins], [0, 256])

        hist_r /= hist_r.sum()
        hist_g /= hist_g.sum()
        hist_b /= hist_b.sum()
        return hist_r, hist_g, hist_b

    def calculate_hsl_histogram(self, image_bgr):
        hls_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
        hist_h = cv2.calcHist([hls_image], [0], None, [self.bins], [0, 180])
        hist_l = cv2.calcHist([hls_image], [1], None, [self.bins], [0, 256])
        hist_s = cv2.calcHist([hls_image], [2], None, [self.bins], [0, 256])

        hist_h /= hist_h.sum()
        hist_l /= hist_l.sum()
        hist_s /= hist_s.sum()
        return hist_h, hist_s, hist_l

    def calculate_similarity(self, hist1, hist2):
        return 1 - cosine(hist1.flatten(), hist2.flatten())

    def compute_feature(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        image_np = np.array(image, dtype=np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        rgb_hist = self.calculate_rgb_histogram(image_bgr)
        hsl_hist = self.calculate_hsl_histogram(image_bgr)
        return (rgb_hist, hsl_hist)

    def find_similar(self, query_hist, best_k=5):
        similarities = []
        query_rgb, query_hsl = query_hist

        self.loader.db.cursor.execute(
            "SELECT image_id, color_histogram FROM images WHERE color_histogram IS NOT NULL;"
        )
        rows = self.loader.db.cursor.fetchall()

        for idx, (image_id, hist_blob) in enumerate(rows):
            db_rgb, db_hsl = pickle.loads(hist_blob)

            rgb_similarity = (
                0.3 * self.calculate_similarity(query_rgb[0], db_rgb[0]) +
                0.59 * self.calculate_similarity(query_rgb[1], db_rgb[1]) +
                0.11 * self.calculate_similarity(query_rgb[2], db_rgb[2])
            )
            hsl_similarity = (
                0.4 * self.calculate_similarity(query_hsl[0], db_hsl[0]) +
                0.3 * self.calculate_similarity(query_hsl[1], db_hsl[1]) +
                0.3 * self.calculate_similarity(query_hsl[2], db_hsl[2])
            )
            compl_similarity = 0.5 * rgb_similarity + 0.5 * hsl_similarity
            similarities.append((image_id, compl_similarity))

            if idx % 100 == 0:
                print(f"Verglichen: {idx} Bilder")

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [img_id for img_id, _ in similarities[:best_k]]
