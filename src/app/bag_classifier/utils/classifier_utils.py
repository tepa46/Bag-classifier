from app.bag_classifier.constants import (
    garbageBagsClassPath,
    paperBagsClassPath,
    plasticBagsClassPath,
    datasetClasses,
)
from app.bag_classifier.utils.images_utils import load_images_from_folder


def collect_images_info():
    """
    Collects information about all images and their classes in the dataset.

    This method combines images from different classes (garbage bags, paper bags, and plastic bags)
        by calling `collect_class_images_info` for each class folder and class name.

    Returns:
        list: A list of tuples (image, image_class)
    """

    images_info = []

    images_info.extend(
        collect_class_images_info(garbageBagsClassPath, datasetClasses[0])
    )
    images_info.extend(collect_class_images_info(paperBagsClassPath, datasetClasses[1]))
    images_info.extend(
        collect_class_images_info(plasticBagsClassPath, datasetClasses[2])
    )

    return images_info


def collect_class_images_info(images_class_folder, images_class_name):
    """
    Collects information about images of class [images_class_name] from a specific class folder.

    This method loads all images from a given folder and associates them with the provided class name.

    Parameters:
        images_class_folder (str): The path to the folder containing the images of a specific class.
        images_class_name (str): The name of the class that the images belong to.

    Returns:
        list: A list of tuples (image, images_class_name)
    """

    images_info = []

    class_images = load_images_from_folder(images_class_folder)

    for class_image in class_images:
        images_info.append((class_image, images_class_name))

    return images_info
