import hnswlib
import numpy as np

class HNSWIndex:
    def __init__(self, d, space='cosine', M=16, efc=200, ef=100):
        self.d = d
        self.space = space
        self.M = M
        self.efc = efc
        self.ef = ef
        self.index = hnswlib.Index(space=space, dim=d)

    def build(self, X, ids):
        """
        Build the HNSW index with the provided data.
        
        :param X: The data points to index, shape (n_samples, d).
        :param ids: The unique identifiers for each data point.
        """
        self.index.init_index(max_elements= X.shape[0], ef_construction=self.efc, M=self.M)
        self.index.add_items(X, ids.astype(np.int64))
        self.index.set_ef(self.ef)

    def save(self, path):
        """
        Save the index to a file.
        
        :param path: The file path where the index will be saved.
        """
        self.index.save_index(path)

    def load(self, path):
        """
        Load the index from a file.
        
        :param path: The file path from which the index will be loaded.
        """
        self.index.load_index(path)
        self.index.set_ef(self.ef)

    def search(self, q, k=5):
        """
        Search for the k nearest neighbors of the query vector. 
        :param q: The query vector to search for.
        :param k: The number of nearest neighbors to return.
        :return: A tuple of (ids, distances) for the k nearest neighbors.
        """
        labels, distances = self.index.knn_query(q.reshape(1, -1).astype(np.float32), k=k)
        return labels[0], distances[0]