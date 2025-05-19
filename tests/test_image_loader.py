from image_recommender.database import Database
from image_recommender.image_loader import ImageLoader

def test_load_image():
    db = Database("images.db")
    loader = ImageLoader(db, base_dir="/Volumes/BigData03/data")

    ids = db.get_all_image_ids()
    if ids:
        img = loader.load_image(ids[0])
        assert img is not None
        assert img.mode == "RGB"
    db.close()

def test_load_image_by_path():
    loader = ImageLoader(None, base_dir="/invalid/path")
    img = loader.load_image_by_path("/invalid/path/image.jpg")
    assert img is None

'''
use: 
cd ImageRecommender/
PYTHONPATH=. pytest tests/
'''