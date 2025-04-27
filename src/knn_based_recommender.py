import numpy as np
from sklearn.neighbors import NearestNeighbors

class KNNRecommender:
    def __init__(self, embeddings, n_neighbors=5):
        self.embeddings = embeddings
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')

    def fit(self):
        self.model.fit(self.embeddings)

    def recommend(self, node_index):
        distances, indices = self.model.kneighbors([self.embeddings[node_index]])
        return indices[0], distances[0]