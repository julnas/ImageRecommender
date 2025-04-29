#implement the color similarity metric
#imports
import numpy as np

#code

def get_histogram(self, pixellist, color):
    'This method initialises an array that counts all the values of one color to be used in a histogramm'
    rgb_array = np.zeros(256)

    for pixel in pixellist:
        rgb_array[pixel[color]] += 1 #der R-Wert der der Pixel hat wird um 1 erhöht

def normalise_histogram(self, histogram, pixel_count):
    normalised_value = histogram / pixel_count
    return normalised_value


def calculate_similarity(self, histograms):
    histogram_picture1, histogram_picture2 = histograms
    similarity_sum = 0
    i = 0

    while i <= 255:
        similarity_sum += (histogram_picture1[i] - histogram_picture2[i]) ** 2
        i += 1

    similarity = 1 / (1+ similarity_sum)

    return similarity

def complete_similarity(self, picture1, picture2):
    histogram_red1 = get_histogram(picture1, 0)
    histogram_red2 = get_histogram(picture2, 0)
    red_histograms = (histogram_red1, histogram_red2)

    histogram_green1 = get_histogram(picture1, 1)
    histogram_green2 = get_histogram(picture2, 1)
    green_histograms = (histogram_green1, histogram_green2)

    histogram_blue1 = get_histogram(picture1, 2)
    histogram_blue2 = get_histogram(picture2, 2)
    blue_histograms = (histogram_blue1, histogram_blue2)

    red_similarity = calculate_similarity(red_histograms)
    green_similarity = calculate_similarity(green_histograms)
    blue_similarity = calculate_similarity(blue_histograms)

    compl_similarity = (0.3 * red_similarity + 0.59 * green_similarity + 0.11 * blue_similarity)

    return compl_similarity

