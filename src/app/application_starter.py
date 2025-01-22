import sys

from PyQt5.QtWidgets import QApplication

from app.bag_classifier.classifier import BagsClassifierInitializer
from app.logging.logger_settings import set_logger_config
from app.view.bag_classifier import ImageClassifierApp


def start_application():
    app = QApplication(sys.argv)
    ex = ImageClassifierApp(BagsClassifierInitializer())
    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    set_logger_config()
    start_application()
