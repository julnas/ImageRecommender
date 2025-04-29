#imports
import numpy as np
import cv2
from scipy.spatial.distance import cosine

# --------------------------------------------------
# Function: calculate_histogram
# Description:
# This function computes color histograms for each channel (Red, Green, Blue)
# of the given image. Each histogram is divided into a specified number of bins 
# (default: 16). After computing, the histograms are normalized so that their sum equals 1.
# This ensures the histograms are comparable across different image sizes.
# --------------------------------------------------
def calculate_histogram(image, bins=16):
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
def calculate_similarity(hist1, hist2):
    similarity = 1 - cosine(hist1.flatten(), hist2.flatten())
    return similarity

# --------------------------------------------------
# Function: complete_similarity
# Description:
# This function calculates the overall color similarity between two images.
# It first computes the color histograms for each image (Red, Green, Blue),
# then measures the similarity per channel using cosine similarity.
# Finally, it combines these similarities using weighted contributions (as used in luminance calculations):
# Red (30%), Green (59%), Blue (11%).
# --------------------------------------------------
def complete_similarity(picture1, picture2, bins=16):
    # Compute histograms for each image
    hist_r1, hist_g1, hist_b1 = calculate_histogram(picture1, bins)
    hist_r2, hist_g2, hist_b2 = calculate_histogram(picture2, bins)
    
    # Calculate cosine similarity for each color channel
    red_similarity = calculate_similarity(hist_r1, hist_r2)
    green_similarity = calculate_similarity(hist_g1, hist_g2)
    blue_similarity = calculate_similarity(hist_b1, hist_b2)
    
    # Combine the similarities using weighted average (standard RGB luminance weights)
    compl_similarity = (0.3 * red_similarity + 0.59 * green_similarity + 0.11 * blue_similarity)
    
    # Print channel-wise similarity (for debugging or insights)
    print(f"Red: {red_similarity}, Green: {green_similarity}, Blue: {blue_similarity}")

    return compl_similarity

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
