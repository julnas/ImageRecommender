import numpy as np
import faiss

class IVFPQIndex:
    def __init__(self, d, nlist=4096, m=16):
        """
        Initialize the IVFPQ index. 
        :param d: The dimensionality of the data points.
        :param nlist: The number of clusters (nlist) for the IVF index.
        :param m: The number of subquantizers for the PQ index.
        """
        quant = faiss.IndexFlatIP(d)
        self.index = faiss.IndexIVFPQ(quant, d, nlist, m, 8)

    def build(self, X, ids):
        """
        Build the IVFPQ index with the provided data.
        
        :param X: The data points to index, shape (n_samples, d).
        :param ids: The unique identifiers for each data point.
        """
        self.index.train(X)
        self.index.add_with_ids(X, ids.astype(np.int64))

    def save(self, path):
        """
        Save the index to a file.
        
        :param path: The file path where the index will be saved.
        """
        faiss.write_index(self.index, path)

    def load(self, path):
        """
        Load the index from a file.
        
        :param path: The file path from which the index will be loaded.
        """
        self.index = faiss.read_index(path)

    def search(self, q, k=5):
        """
        Search for the k nearest neighbors of the query vector. 
        :param q: The query vector to search for.
        :param k: The number of nearest neighbors to return.
        :return: A tuple of (ids, distances) for the k nearest neighbors.
        """
        scores, ids = self.index.search(q.reshape(1, -1).astype(np.float32), k)
        return ids[0], scores[0]