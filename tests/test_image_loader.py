from image_recommender.image_loader import ImageLoader


def test_load_image(image_store):
    loader = ImageLoader(
        image_store["db"],
        base_dir=str(image_store["image_dir"]),
    )

    image = loader.load_image(1)

    assert image is not None
    assert image.mode == "RGB"
    assert image.size == (16, 12)
    assert image.getpixel((0, 0)) == (255, 0, 0)


def test_load_image_by_path(image_store):
    loader = ImageLoader(None, base_dir=str(image_store["image_dir"]))

    image = loader.load_image_by_path(str(image_store["image_dir"] / "green.png"))

    assert image is not None
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (0, 255, 0)


def test_load_image_by_path_returns_none_for_missing_file(image_store):
    loader = ImageLoader(None, base_dir=str(image_store["image_dir"]))

    image = loader.load_image_by_path(str(image_store["image_dir"] / "missing.png"))

    assert image is None
