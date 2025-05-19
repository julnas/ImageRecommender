from image_recommender.database import Database
from image_recommender.image_loader import ImageLoader
from image_recommender.recomender import Recommender
from SimilarityMetrics.color_similarity_metric import ColorSimilarity


def test_recommender_outputs_top_k():
    db = Database("images.db")
    loader = ImageLoader(db, base_dir="/Volumes/BigData03/data")
    color_sim = ColorSimilarity(loader)
    metrics = {"color": color_sim}
    recommender = Recommender(db, loader, metrics)

    ids = db.get_all_image_ids()
    if ids:
        results = recommender.recommend(input_image_id=ids[0], best_k=3)
        assert "color" in results
        assert len(results["color"]) == 3
    db.close()


"""
use: 
cd ImageRecommender/
PYTHONPATH=. pytest tests/
"""
