from image_recommender.database import Database

#testing get_all_image_ids 
def test_get_all_image_ids():
    db = Database("images.db")
    ids = db.get_all_image_ids()
    db.close()
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)

#testing get_image_path
def test_get_image_path():
    db = Database("images.db")
    ids = db.get_all_image_ids()
    if ids:
        path = db.get_image_path(ids[0])
        assert isinstance(path, str)
    db.close()

'''
use: 
cd ImageRecommender/
PYTHONPATH=. pytest tests/
'''