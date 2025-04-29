#implement the color similarity metric
#imports
import numpy as np
import cv2

#code
def reshape_image(image):
    image = image.reshape(-1, 3)
    return image

def get_histogram(pixellist, color):
    'This method initialises an array that counts all the values of one color to be used in a histogramm'
    rgb_array = np.zeros(256)

    for pixel in pixellist:
        rgb_array[pixel[color]] += 1 #der R-Wert der der Pixel hat wird um 1 erhöht

    return rgb_array

def get_pixelcount(image):
    pixel_count = image.shape[0] * image.shape[1]
    return pixel_count

def normalise_histogram(histogram, pixel_count):
    normalised_value = histogram / pixel_count
    return normalised_value


def calculate_similarity(histograms):
    histogram_picture1, histogram_picture2 = histograms
    similarity_sum = 0
    i = 0

    while i <= 255:
        similarity_sum += (histogram_picture1[i] - histogram_picture2[i]) ** 2
        i += 1

    similarity = 1 / (1+ similarity_sum)

    return similarity

def complete_similarity(picture1, picture2):
    pixelcount_image1 = get_pixelcount(picture1)
    pixelcount_image2 = get_pixelcount(picture2)

    histogram_red1 = get_histogram(picture1, 0)
    histogram_red1 =normalise_histogram(histogram_red1, pixelcount_image1)
    histogram_red2 = get_histogram(picture2, 0)
    histogram_red2 = normalise_histogram(histogram_red2, pixelcount_image2)
    red_histograms = (histogram_red1, histogram_red2)

    histogram_green1 = get_histogram(picture1, 1)
    histogram_green1 = normalise_histogram(histogram_green1, pixelcount_image1)
    histogram_green2 = get_histogram(picture2, 1)
    histogram_green2 = normalise_histogram(histogram_green2, pixelcount_image2)
    green_histograms = (histogram_green1, histogram_green2)

    histogram_blue1 = get_histogram(picture1, 2)
    histogram_blue1 =normalise_histogram(histogram_blue1, pixelcount_image1)
    histogram_blue2 = get_histogram(picture2, 2)
    histogram_blue2 = normalise_histogram(histogram_blue2, pixelcount_image2)
    blue_histograms = (histogram_blue1, histogram_blue2)

    red_similarity = calculate_similarity(red_histograms)
    green_similarity = calculate_similarity(green_histograms)
    blue_similarity = calculate_similarity(blue_histograms)

    compl_similarity = (0.3 * red_similarity + 0.59 * green_similarity + 0.11 * blue_similarity)

    return compl_similarity


filename1 = "/Users/jule/Documents/Uni/4. Semester/Big Data Engineering/Testbilder-20250328/dove-2516641_1920.jpg"
filename2 = '/Users/jule/Documents/Uni/4. Semester/Big Data Engineering/Testbilder-20250328/hummingbird-2139278_1920.jpg'

image1 = cv2.imread(filename1)
