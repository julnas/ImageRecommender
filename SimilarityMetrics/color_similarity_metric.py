#implement the color similarity metric
#imports
import numpy as np
import cv2
from scipy.spatial.distance import cosine

#code
def calculate_histogram(image, bins=16):
    # Erstelle ein Array für die Histogramme, mit der Anzahl der Bins
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])

    # Normalisieren der Histogramme
    hist_r /= hist_r.sum()
    hist_g /= hist_g.sum()
    hist_b /= hist_b.sum()

    return hist_r, hist_g, hist_b

def calculate_similarity(hist1, hist2):
    similarity = 1 - cosine(hist1.flatten(), hist2.flatten())
    return similarity

def complete_similarity(picture1, picture2, bins=16):
    # Berechne die Histogramme effizient
    hist_r1, hist_g1, hist_b1 = calculate_histogram(picture1, bins)
    hist_r2, hist_g2, hist_b2 = calculate_histogram(picture2, bins)
    
    # Berechne die Ähnlichkeit für jeden Kanal
    red_similarity = calculate_similarity(hist_r1, hist_r2)
    green_similarity = calculate_similarity(hist_g1, hist_g2)
    blue_similarity = calculate_similarity(hist_b1, hist_b2)
    
    # Gesamtähnlichkeit berechnen (mit den typischen Gewichtungen für RGB)
    compl_similarity = (0.3 * red_similarity + 0.59 * green_similarity + 0.11 * blue_similarity)
    print(f"Red: {red_similarity}, Green: {green_similarity}, Blue: {blue_similarity}")

    return compl_similarity


filename1 = "/Users/jule/Downloads/Elbe_-_flussaufwärts_kurz_nach_Ort_Königstein.jpg"
filename2 = '/Users/jule/Downloads/red-background.png'

image1 = cv2.imread(filename1)
image2 = cv2.imread(filename2)

similarity = complete_similarity(image1, image2, bins=16)
print("The similarity between image 1 and 2 is:", similarity)