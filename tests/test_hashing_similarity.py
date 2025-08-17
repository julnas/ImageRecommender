from PIL import Image
from SimilarityMetrics.hashing_similarity_metric import HashingSimilarity

def test_hash_similarity_identity():
    img = Image.new("RGB", (16,16), (0,0,0))
    metric = HashingSimilarity(loader=None)
    h = metric.compute_feature(img)
    s = metric._similarity(h, h)
    assert abs(s - 1.0) < 1e-6
