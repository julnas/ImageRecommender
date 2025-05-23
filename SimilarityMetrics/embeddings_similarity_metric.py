# implement the embeddings similarity metric
#imports 
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image
from scipy.spatial.distance import cosine
import pickle

class EmbeddingSimilarity:
    def __init__(self, device=None):
        """
        Initializes the EmbeddingSimilarity class with a pre-trained model.

        Parameters:
        model: A pre-trained model for generating image embeddings.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ResNet18 laden, letzten FC-Layer entfernen (letztes Modul wegnehmen)
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # ohne fc
        self.model.to(self.device)
        self.model.eval()

        # Transformation passend für ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def compute_feature(self, image, db=None):
        if isinstance(image, str):
            print("Image is a string")
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            print("Image is a numpy array")
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_t = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img_t)  # shape: (1, 512, 1, 1)
        embedding = embedding.squeeze().cpu().numpy()
        return embedding
    
    # --------------------------------------------------
    # Function: calculate_similarity
    # Description:
    # This function calculates the similarity between two histograms using cosine similarity.
    # Cosine similarity measures the cosine of the angle between two non-zero vectors.
    # It returns a value between -1 (completely different) and 1 (identical).
    # --------------------------------------------------
    def calculate_similarity(self, embedding1, embedding2):
        similarity = 1 - cosine(embedding1.flatten(), embedding2.flatten())
        return similarity
    
    # --------------------------------------------------
    # Function: find_similar
    # Description:
    # This function calculates the overall color similarity and IDs for every Image.
    # it appendds the id and the similarity to a list and sorts it.
    # The top-k most similar images are returned.
    # --------------------------------------------------
    def find_similar(self, embedding, best_k=5):
        similarities = []

        # get all images with embeddings from the database
        self.loader.db.cursor.execute(
            "SELECT image_id, embedding FROM images WHERE embedding IS NOT NULL;"
        )
        rows = self.loader.db.cursor.fetchall()

        for idx, (image_id, hist_blob) in enumerate(rows):
            db_hist = pickle.loads(hist_blob)  # unpickle the histogram

            # Calculate cosine similarity for each color channel
            compl_similarity = self.calculate_similarity(embedding, db_hist)

            # append every image id and the similarity to a list, so we can compsare them later
            similarities.append((image_id, compl_similarity))

            # counting the pictures
            if idx % 100 == 0:
                print(f"Compared: {idx} piktures")

        # Sort descending by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [img_id for img_id, _ in similarities[:best_k]]
    

"""# Dateipfade
filename1 = "/Users/jule/Downloads/Eiffelturm-Paris2-scaled.jpg"
filename2 = "/Users/jule/Downloads/eiffelturm-unteransicht.jpg"

recommender = EmbeddingSimilarity()
image1 = cv2.imread(filename1)
image2 = cv2.imread(filename2)

# Feature-Vektoren berechnen
image1_feature = recommender.compute_feature(image1)
image2_feature = recommender.compute_feature(image2)

# Kosinus-Ähnlichkeit
similarity = np.dot(image1_feature, image2_feature) / (np.linalg.norm(image1_feature) * np.linalg.norm(image2_feature))
print("The similarity between image 1 and 2 is:", similarity)"""
