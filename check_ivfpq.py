import faiss
import numpy as np

index = faiss.read_index("indexes/emb_ivfpq.faiss")
print("dim:", index.d, "ntotal:", index.ntotal)

q = np.random.rand(1, index.d).astype(np.float32)
print(index.search(q, 5))
