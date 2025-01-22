from abc import abstractmethod, ABCMeta

from PyQt5.QtCore import QThread, pyqtSignal


class AbstractQThreadMeta(type(QThread), ABCMeta):
    pass


class Classifier(QThread, metaclass=AbstractQThreadMeta):
    """
    Abstract base class for implementing different types of classifiers.

    Attributes:
        classifier_image_path (str):
            The file path of the image to be classified.
            !!! Set [classifier_image_path] before call [classify_image] method !!!
        classifier_answer (str):
            The result of the classification in a human-readable format.
        in_progress (PYQT_SIGNAL):
            Indicates whether the classification task is currently running.
        completed (PYQT_SIGNAL):
            Indicates whether the classification process is completed.
    """

    in_progress = pyqtSignal()
    completed = pyqtSignal()

    classifier_image_path: str = None
    classifier_answer: str

    def run(self):
        """
        Executes the classification process for the image in a separate thread.
        """
        self.in_progress.emit()
        self.classifier_answer = None
        self.classify_image()
        self.completed.emit()

    @abstractmethod
    def classify_image(self):
        """
        Classify a single image.

        !!! This method must assign the classification information to a [classifier_answer] variable !!!
        """
        pass

    def set_image_to_classify(self, image_path):
        """
        Set [classifier_image_path] for future classification.

        :param image_path: The file path of the image
        """
        self.classifier_image_path = image_path


class ClassifierInitializer(QThread, metaclass=AbstractQThreadMeta):
    """
    Abstract base class for initializing and training classifiers.

    Attributes:
        classifier (Classifier):
            The `Classifier` instance that will be initialized and trained.
        in_progress (PYQT_SIGNAL):
            Indicates whether the initialization task is currently running.
        completed (PYQT_SIGNAL):
            Indicates whether the initialization process is completed.
    """

    in_progress = pyqtSignal()
    completed = pyqtSignal()

    classifier: Classifier = None

    def run(self):
        """
        Executes the initialization process for the classifier in a separate thread.
        """

        self.in_progress.emit()
        self.initialize_classifier()
        self.completed.emit()

    @abstractmethod
    def initialize_classifier(self):
        """
        Initialize and train classifier.

        !!! This method must assign an instance of initialized and trained classifier to the [classifier] attribute !!!
        """
        pass
