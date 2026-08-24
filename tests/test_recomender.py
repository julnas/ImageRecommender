import pickle

from image_recommender.image_loader import ImageLoader
from image_recommender.recomender import Recommender
from SimilarityMetrics.color_similarity_metric import ColorSimilarity


def test_recommender_outputs_top_k(image_store):
    db = image_store["db"]
    loader = ImageLoader(db, base_dir=str(image_store["image_dir"]))
    color_metric = ColorSimilarity(loader, bins=8)

    for image_id in image_store["image_ids"]:
        image = loader.load_image(image_id)
        feature_blob = pickle.dumps(color_metric.compute_feature(image))
        db.cursor.execute(
            "UPDATE images SET color_histogram = ? WHERE image_id = ?",
            (feature_blob, image_id),
        )

    db.connection.commit()

    metrics = {"color": color_metric}
    recommender = Recommender(db, loader, metrics)
    query_image = loader.load_image(1)

    results = recommender.recommend(query_image, best_k=2)

    assert set(results) == {"color"}
    assert len(results["color"]) == 2
    assert results["color"][0] == 1
    assert recommender.last_timings["color"]["total_seconds"] >= 0
