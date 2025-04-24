#implement the color similarity metric
#imports
import numpy as np

#code
class Histograms():
    def get_histogram(self, pixellist, color):
        'This method initialises an array that counts all the values of one color to be used in a histogramm'
        rgb_array = np.zeros(256)

        for pixel in pixellist:
            rgb_array[pixel[color]] += 1 #der R-Wert der der Pixel hat wird um 1 erhöht

    def normalise_histogram(self, histogram, pixel_count):
        normalised_value = histogram / pixel_count
        return normalised_value
    
class RGB_Similarity():
    def calculate_complete_similarity(self):
        pass

    def calculate_red_similarity(self):
        pass

    def calculate_green_similarity(self):
        pass

    def calculate_blue_similarity(self):
        pass
