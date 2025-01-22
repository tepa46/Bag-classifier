import os

from kaggle import KaggleApi

from src.app.bag_classifier.constants import datasetKagglePath, datasetSavePath


def is_dataset_loaded():
    """
    Checks if the dataset has already been downloaded and saved to the specified path.
    """

    if os.path.exists(datasetSavePath):
        return True
    return False


def load_dataset():
    """
    Downloads and save to the specified path the dataset if it is not already downloaded.
    """

    if is_dataset_loaded():
        return

    api = KaggleApi()
    api.authenticate()

    os.makedirs(datasetSavePath, exist_ok=True)

    api.dataset_download_files(datasetKagglePath, path=datasetSavePath, unzip=True)
