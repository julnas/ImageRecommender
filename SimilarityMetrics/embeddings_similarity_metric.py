import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from scipy.spatial.distance import cosine
import pickle

class EmbeddingSimilarity:
    def __init__(self, loader, device=None):
        self.loader = loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def compute_feature(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        img_t = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img_t)
        embedding = embedding.squeeze().cpu().numpy()
        return embedding

    def calculate_similarity(self, emb1, emb2):
        return 1 - cosine(emb1.flatten(), emb2.flatten())

    def find_similar(self, embedding, best_k=5):
        similarities = []

        self.loader.db.cursor.execute(
            "SELECT image_id, embedding FROM images WHERE embedding IS NOT NULL;"
        )
        rows = self.loader.db.cursor.fetchall()

        for idx, (image_id, emb_blob) in enumerate(rows):
            db_emb = pickle.loads(emb_blob)
            sim = self.calculate_similarity(embedding, db_emb)
            similarities.append((image_id, sim))

            if idx % 100 == 0:
                print(f"Verglichen: {idx} Bilder")

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [img_id for img_id, _ in similarities[:best_k]]
