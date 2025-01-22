import logging

import numpy as np

from app.bag_classifier.constants import garbageBagsClassPath, paperBagsClassPath, \
    plasticBagsClassPath
from app.bag_classifier.hypotheses.hypotheses import get_hypothesis_1_feature_value, get_hypothesis_2_feature_value, \
    get_hypothesis_5_feature_value, get_hypothesis_4_feature_value, get_hypothesis_3_feature_value, \
    get_hypothesis_6_feature_value, get_hypothesis_7_feature_value, get_hypothesis_9_feature_value, get_hypothesis_8_feature_value, \
    get_hypothesis_10_feature_value
from app.bag_classifier.hypotheses.hypotheses_tests import utest, kstest
from app.bag_classifier.utils.images_utils import load_images_from_folder

logger = logging.getLogger(__name__)


def get_features(paths, feature_fun):
    """
    Extracts features from a set of images located in the given paths using the specified feature function.

    Returns:
        list: A list containing the extracted features for all images from the specified paths.
    """

    bags_statistic = []
    for bags_path in paths:
        images = load_images_from_folder(bags_path)
        bags_statistic.extend([feature_fun(img) for img in images])

    return bags_statistic


def verify_hypothesis(this_paths, other_paths, feature_fun):
    """
    Compares the features extracted from two sets of images using statistical tests:
        - Mann–Whitney U-test
        - Kolmogorov–Smirnov test

    Parameters:
        this_paths (list[str]): A list of paths to directories containing images for the first set.
        other_paths (list[str]): A list of paths to directories containing images for the second set.
        feature_fun ((image) -> feature_value): A function that takes an image as input and returns a feature value.
    """

    this_bags_features = get_features(this_paths, feature_fun)
    logger.info("THIS")
    logger.info(f"median {np.median(this_bags_features)}")
    logger.info(f"mean {np.mean(this_bags_features)}")
    logger.info(f"min {np.min(this_bags_features)}")
    logger.info(f"max {np.max(this_bags_features)}")

    other_bags_features = get_features(other_paths, feature_fun)
    logger.info("OTHER")
    logger.info(f"median {np.median(other_bags_features)}")
    logger.info(f"mean {np.mean(other_bags_features)}")
    logger.info(f"min {np.min(other_bags_features)}")
    logger.info(f"max {np.max(other_bags_features)}")

    u_res = utest(this_bags_features, other_bags_features)

    ks_res = kstest(this_bags_features, other_bags_features)

    results = {
        "u-test": u_res,
        "ks-test": ks_res,
    }

    for test, result in results.items():
        logger.info(f"{test}: Statistic = {result}")


# Мусорные пакеты имеют темные цвета
def verify_hypothesis_1():
    logger.info("$Hypotheses verification hypotheses 1 verification")
    this_bags_paths = [garbageBagsClassPath]
    other_bags_paths = [paperBagsClassPath, plasticBagsClassPath]

    verify_hypothesis(this_bags_paths, other_bags_paths, get_hypothesis_1_feature_value)


# Пластиковые пакеты и мусорные пакеты часто содержат яркие блики
def verify_hypothesis_2():
    logger.info("$Hypotheses verification hypotheses 2 verification")
    this_bags_paths = [garbageBagsClassPath, plasticBagsClassPath]
    other_bags_paths = [paperBagsClassPath]

    verify_hypothesis(this_bags_paths, other_bags_paths, get_hypothesis_2_feature_value)


# На изображениях с бумажными пакетами много длинных отрезков
def verify_hypothesis_3():
    logger.info("$Hypotheses verification hypotheses 3 verification")
    this_bags_paths = [paperBagsClassPath]
    other_bags_paths = [garbageBagsClassPath, plasticBagsClassPath]

    verify_hypothesis(this_bags_paths, other_bags_paths, get_hypothesis_3_feature_value)


# Бумажные пакеты имеют более насыщенные цвета
def verify_hypothesis_4():
    logger.info("$Hypotheses verification hypotheses 4 verification")
    this_bags_paths = [paperBagsClassPath]
    other_bags_paths = [garbageBagsClassPath, plasticBagsClassPath]

    verify_hypothesis(this_bags_paths, other_bags_paths, get_hypothesis_4_feature_value)


# Бумажные пакеты часто имеют светлокоричневый цвет
def verify_hypothesis_5():
    logger.info("$Hypotheses verification hypotheses 5 verification")
    this_bags_paths = [paperBagsClassPath]
    other_bags_paths = [garbageBagsClassPath, plasticBagsClassPath]

    verify_hypothesis(this_bags_paths, other_bags_paths, get_hypothesis_5_feature_value)


# Бумажные пакеты --- матовые
def verify_hypothesis_6():
    logger.info("$Hypotheses verification hypotheses 6 verification")
    this_bags_paths = [paperBagsClassPath]
    other_bags_paths = [garbageBagsClassPath, plasticBagsClassPath]

    verify_hypothesis(this_bags_paths, other_bags_paths, get_hypothesis_6_feature_value)


# Пластиковые пакеты имеют яркие цвета
def verify_hypothesis_7():
    logger.info("$Hypotheses verification hypotheses 7 verification")
    this_bags_paths = [plasticBagsClassPath]
    other_bags_paths = [garbageBagsClassPath, paperBagsClassPath]

    verify_hypothesis(this_bags_paths, other_bags_paths, get_hypothesis_7_feature_value)


# Из-за сильно выраженных складок на мусорных пакетах, найденные контуры мусорных пакетов по площади будут меньше,
# чем контуры пластиковых и бумажных пакетов
def verify_hypothesis_8():
    logger.info("$Hypotheses verification hypotheses 8 verification")
    this_bags_paths = [garbageBagsClassPath]
    other_bags_paths = [plasticBagsClassPath, paperBagsClassPath]

    verify_hypothesis(this_bags_paths, other_bags_paths, get_hypothesis_8_feature_value)


# Контуры бумажных и пластиковых пакетов часто имеют меньше углов
def verify_hypothesis_9():
    logger.info("$Hypotheses verification hypotheses 9 verification")
    this_bags_paths = [plasticBagsClassPath, paperBagsClassPath]
    other_bags_paths = [garbageBagsClassPath]

    verify_hypothesis(this_bags_paths, other_bags_paths, get_hypothesis_9_feature_value)


# Пластиковые пакеты из-за своей прозрачности могут иметь участки ненасыщенного цвета
def verify_hypothesis_10():
    logger.info("$Hypotheses verification hypotheses 10 verification")
    this_bags_paths = [plasticBagsClassPath]
    other_bags_paths = [garbageBagsClassPath, paperBagsClassPath]

    verify_hypothesis(this_bags_paths, other_bags_paths, get_hypothesis_10_feature_value)
