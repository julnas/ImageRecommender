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
    def calculate_rgb_histogram(self, image, bins=16):
        # Calculate histograms for Red, Green, and Blue channels
        hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])

        # Normalize histograms so that the sum equals 1
        hist_r /= hist_r.sum()
        hist_g /= hist_g.sum()
        hist_b /= hist_b.sum()

        return hist_r, hist_g, hist_b
    
    def calculate_hsl_histogram(self, image, bins=16):
        # convert BGR to HLS (OpenCV uses HLS not HSL – Reihenfolge: H, L, S)
        hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        hist_h = cv2.calcHist([hls_image], [0], None, [bins], [0, 180])  # Hue in OpenCV: 0–179
        hist_l = cv2.calcHist([hls_image], [1], None, [bins], [0, 256])
        hist_s = cv2.calcHist([hls_image], [2], None, [bins], [0, 256])

        hist_h /= hist_h.sum()
        hist_l /= hist_l.sum()
        hist_s /= hist_s.sum()

        return hist_h, hist_s, hist_l

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
        rgb_hist = self.calculate_histogram(image_bgr, bins=self.bins)
        hsl_hist = self.calculate_hsl_histogram(image_bgr, bins=self.bins)
        return rgb_hist, hsl_hist

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

        # Entpacke Query-Histogramm
        query_rgb, query_hsl = query_hist

        # Lade alle gespeicherten Histogramme aus der Datenbank
        self.loader.db.cursor.execute(
            "SELECT image_id, color_histogram FROM images WHERE color_histogram IS NOT NULL;"
        )
        rows = self.loader.db.cursor.fetchall()

        for idx, (image_id, hist_blob) in enumerate(rows):
            # Entpacke das RGB- und HSL-Histogramm
            db_rgb, db_hsl = pickle.loads(hist_blob)

            # RGB-Ähnlichkeiten berechnen
            red_similarity = self.calculate_similarity(query_rgb[0], db_rgb[0])
            green_similarity = self.calculate_similarity(query_rgb[1], db_rgb[1])
            blue_similarity = self.calculate_similarity(query_rgb[2], db_rgb[2])
            rgb_similarity = 0.3 * red_similarity + 0.59 * green_similarity + 0.11 * blue_similarity

            # HSL-Ähnlichkeiten berechnen (H, S, L)
            h_similarity = self.calculate_similarity(query_hsl[0], db_hsl[0])
            s_similarity = self.calculate_similarity(query_hsl[1], db_hsl[1])
            l_similarity = self.calculate_similarity(query_hsl[2], db_hsl[2])
            hsl_similarity = 0.4 * h_similarity + 0.3 * s_similarity + 0.3 * l_similarity

            # Kombinierte Gesamtähnlichkeit
            compl_similarity = 0.5 * rgb_similarity + 0.5 * hsl_similarity

            # Bild-ID und Ähnlichkeit speichern
            similarities.append((image_id, compl_similarity))

            # Optional: Fortschritt anzeigen
            if idx % 100 == 0:
                print(f"Verglichen: {idx} Bilder")

        # Nach Ähnlichkeit sortieren (absteigend)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Top-k Bild-IDs zurückgeben
        return [img_id for img_id, _ in similarities[:best_k]]




# --------------------------------------------------
# Import & Instanz
# --------------------------------------------------
from PIL import Image

# Lade die Bilder mit PIL (für Kompatibilität)
image1 = Image.open("/Users/jule/Downloads/Elbe_-_flussaufwärts_kurz_nach_Ort_Königstein.jpg").convert("RGB")
image2 = Image.open("/Users/jule/Downloads/red-background.png").convert("RGB")

# Instanz der Farbsimilaritätsklasse
color_sim = ColorSimilarity(loader=None, bins=16)

# --------------------------------------------------
# Features berechnen (RGB + HSL)
# --------------------------------------------------

# compute_feature muss jetzt beide Histogrammtypen liefern
def compute_combined_feature(image):
    # RGB-Feature
    image_np = np.array(image)
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    rgb_hist = color_sim.calculate_histogram(image_bgr, bins=color_sim.bins)
    hsl_hist = color_sim.calculate_hsl_histogram(image_bgr, bins=color_sim.bins)
    return (rgb_hist, hsl_hist)

# Berechne Features
feature1 = compute_combined_feature(image1)
feature2 = compute_combined_feature(image2)

# --------------------------------------------------
# Ähnlichkeit berechnen (RGB + HSL kombiniert)
# --------------------------------------------------

# RGB-Vergleich
r1, g1, b1 = feature1[0]
r2, g2, b2 = feature2[0]
rgb_sim = (
    0.3 * color_sim.calculate_similarity(r1, r2)
    + 0.59 * color_sim.calculate_similarity(g1, g2)
    + 0.11 * color_sim.calculate_similarity(b1, b2)
)

# HSL-Vergleich
h1, s1, l1 = feature1[1]
h2, s2, l2 = feature2[1]
hsl_sim = (
    0.4 * color_sim.calculate_similarity(h1, h2)
    + 0.3 * color_sim.calculate_similarity(s1, s2)
    + 0.3 * color_sim.calculate_similarity(l1, l2)
)

# Gesamtscore
final_similarity = 0.5 * rgb_sim + 0.5 * hsl_sim

# --------------------------------------------------
# Ausgabe
# --------------------------------------------------
print("Ähnlichkeit zwischen Bild 1 und Bild 2:", final_similarity)

