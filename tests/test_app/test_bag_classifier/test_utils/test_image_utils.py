import pytest
from unittest.mock import patch, MagicMock
import os
import cv2
from app.bag_classifier.utils.images_utils import load_images_from_folder, load_image_from_filepath


@patch("app.bag_classifier.utils.images_utils.cv2.imread")
def test_load_image_from_filepath(mock_imread):
    filepath = "/path/to/image.jpg"

    mock_imread.return_value = MagicMock()

    result = load_image_from_filepath(filepath)

    mock_imread.assert_called_once_with(filepath)
    assert result is not None

    mock_imread.return_value = None

    result = load_image_from_filepath(filepath)
    assert result is None


@patch("app.bag_classifier.utils.images_utils.load_image_from_filepath")
def test_load_images_from_folder(mock_load_image_from_filepath):
    folder = "/path/to/folder"

    result = load_images_from_folder(folder)
    assert result == []

    mock_listdir.return_value = ["image1.jpg", "image2.jpg", "image3.jpg"]
    mock_load_image_from_filepath.side_effect = [
        MagicMock(),  # image1.jpg
        None,         # image2.jpg (не удалось загрузить)
        MagicMock(),  # image3.jpg
    ]

    result = load_images_from_folder(folder)

    assert len(result) == 2
    mock_load_image_from_filepath.assert_any_call(os.path.join(folder, "image1.jpg"))
    mock_load_image_from_filepath.assert_any_call(os.path.join(folder, "image2.jpg"))
    mock_load_image_from_filepath.assert_any_call(os.path.join(folder, "image3.jpg"))