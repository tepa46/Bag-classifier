import logging

import numpy as np

from scipy import stats

from app.bag_classifier.constants import alpha

logger = logging.getLogger(__name__)


def check_test_p_value(p_value):
    if p_value < alpha:
        return True

    return False


def utest(this_data: np.typing.ArrayLike, other_data: np.typing.ArrayLike):
    """
    Mann–Whitney U-test

    !Does not require a normal distribution!
    """

    stat, p_value = stats.mannwhitneyu(this_data, other_data)

    logger.info(f"utest result: stat {stat}, p_value {p_value}")
    return check_test_p_value(p_value)


def kstest(this_data: np.typing.ArrayLike, other_data: np.typing.ArrayLike):
    """
    Kolmogorov–Smirnov test

    !Does not require a normal distribution!
    """
    stat, p_value = stats.ks_2samp(this_data, other_data)

    logger.info(f"ktest result: result: stat {stat}, p_value {p_value}")
    return check_test_p_value(p_value)
