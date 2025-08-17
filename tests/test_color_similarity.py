import numpy as np
from PIL import Image
from SimilarityMetrics.color_similarity_metric import ColorSimilarity

def test_color_compute_feature_and_vec():
    img = Image.new("RGB", (16, 16), (255, 0, 0))
    metric = ColorSimilarity(loader=None, bins=8)
    feat = metric.compute_feature(img)
    v = metric._feature_to_vec(feat)
    assert v.ndim == 1
    assert np.isfinite(v).all()

def test_color_similarity_self_vs_other():
    img1 = Image.new("RGB", (16, 16), (255, 0, 0))
    img2 = Image.new("RGB", (16, 16), (0, 255, 0))
    metric = ColorSimilarity(loader=None, bins=8)
    f1 = metric.compute_feature(img1)
    f2 = metric.compute_feature(img2)
    v1, v2 = metric._feature_to_vec(f1), metric._feature_to_vec(f2)
    # Ähnlichkeit mit sich selbst > Ähnlichkeit mit anderem Bild
    sim_self = np.dot(v1, v1)
    sim_other = np.dot(v1, v2)
    assert sim_self >= sim_other
