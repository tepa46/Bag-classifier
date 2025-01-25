import pytest
from unittest.mock import patch, MagicMock
import os
import cv2
from app.bag_classifier.utils.images_utils import (
    load_images_from_folder,
    load_image_from_filepath,
)


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
