def test_get_all_image_ids(image_store):
    db = image_store["db"]
    ids = db.get_all_image_ids()

    assert len(ids) == 3
    assert set(ids) == {1, 2, 3}
    assert all(isinstance(image_id, int) for image_id in ids)


def test_get_image_path(image_store):
    db = image_store["db"]

    assert db.get_image_path(1) == "red.png"
    assert db.get_image_path(999) is None
