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

    def compute_feature(self, image, db):
        """
        Computes the feature vector for the input image using the pre-trained model.

        Parameters:
        image: The input image to be processed.

        Returns:
        feature_vector: The computed feature vector for the input image.
        """
        # Extract embeddings
        img_t = self.transform(image).unsqueeze(0).to(self.device)  # batch dim
        with torch.no_grad():
            embedding = self.model(img_t)  # shape: (1, 512, 1, 1)
        embedding = embedding.squeeze().cpu().numpy()  # shape: (512,)
        return embedding
