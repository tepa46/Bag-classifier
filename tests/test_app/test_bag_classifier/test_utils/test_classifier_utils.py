import pytest
from unittest.mock import patch

from app.bag_classifier.utils.classifier_utils import collect_images_info, collect_class_images_info
from app.bag_classifier.constants import garbageBagsClassPath, paperBagsClassPath, plasticBagsClassPath, datasetClasses




@patch("app.bag_classifier.utils.images_utils.load_images_from_folder")
def test_collect_class_images_info(mock_load_images_from_folder):
    mock_load_images_from_folder.return_value = ["image1", "image2"]

    images_class_folder = "test_folder"
    images_class_name = "test_class"

    result = collect_class_images_info(images_class_folder, images_class_name)

    assert result == [("image1", "test_class"), ("image2", "test_class")]

    mock_load_images_from_folder.assert_called_once_with(images_class_folder)


@patch("app.bag_classifier.utils.classifier_utils.collect_class_images_info")
def test_collect_images_info(mock_collect_class_images_info):
    mock_collect_class_images_info.side_effect = [
        [("garbage1.jpg", "garbage")],
        [("paper1.jpg", "paper")],
        [("plastic1.jpg", "plastic")],
    ]

    result = collect_images_info()

    assert result == [
        ("garbage1.jpg", "garbage"),
        ("paper1.jpg", "paper"),
        ("plastic1.jpg", "plastic"),
    ]
    mock_collect_class_images_info.assert_any_call(garbageBagsClassPath, datasetClasses[0])
    mock_collect_class_images_info.assert_any_call(paperBagsClassPath, datasetClasses[1])
    mock_collect_class_images_info.assert_any_call(plasticBagsClassPath, datasetClasses[2])
    assert mock_collect_class_images_info.call_count == 3
