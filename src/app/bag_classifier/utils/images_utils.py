import os
import cv2


def load_images_from_folder(folder):
    """
    Loads all images from the specified folder.

    Parameters:
        folder (str): The path to the folder from which images need to be loaded.

    Returns:
        list: A list of images loaded from the specified folder.
    """

    images = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        image = load_image_from_filepath(filepath)
        if image is not None:
            images.append(image)
    return images


def load_image_from_filepath(filepath):
    """
    Loads an image from the file at the specified path.
    Uses the OpenCV library to read the image from disk.

    Parameters:
        filepath (str): The path to the image file.

    Returns:
        The image loaded from the specified path, or `None` if the image could not be loaded.
    """

    return cv2.imread(filepath)
