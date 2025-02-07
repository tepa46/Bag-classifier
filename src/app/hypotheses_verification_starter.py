from app.bag_classifier.hypotheses.hypotheses_verification import (
    verify_hypothesis_5,
    verify_hypothesis_4,
    verify_hypothesis_3,
    verify_hypothesis_2,
    verify_hypothesis_1,
    verify_hypothesis_6,
    verify_hypothesis_7,
    verify_hypothesis_9,
    verify_hypothesis_10,
    verify_hypothesis_8,
)
from app.loader.dataset_loader import load_dataset
from app.logging.logger_settings import set_logger_config


def start_verification():
    verify_hypothesis_1()
    verify_hypothesis_2()
    verify_hypothesis_3()
    verify_hypothesis_4()
    verify_hypothesis_5()
    verify_hypothesis_6()
    verify_hypothesis_7()
    verify_hypothesis_8()
    verify_hypothesis_9()
    verify_hypothesis_10()


if __name__ == "__main__":
    set_logger_config()
    load_dataset()
    start_verification()
