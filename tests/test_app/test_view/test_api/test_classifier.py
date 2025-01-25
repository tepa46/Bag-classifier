import pytest
from pytestqt import qtbot
from abc import abstractmethod
from app.view.api.classifier import Classifier, ClassifierInitializer

class TestClassifier:
    @abstractmethod
    def classifier(self) -> Classifier:
        raise NotImplementedError("You must implement the Classifier fixture")

    def test_running(self, qtbot):
        try:
            classifier = self.classifier()
        except NotImplementedError:
            return
        
        self.check_in_progress = False
        def in_progress():
            self.check_in_progress = True
        
        self.check_completed = False
        def completed():
            self.check_completed = True

        classifier.in_progress.connect(in_progress)
        classifier.completed.connect(completed)

        with qtbot.waitSignal(classifier.finished, timeout=300):
            classifier.start()

        assert self.check_in_progress
        assert self.check_in_progress
        assert classifier.classifier_answer is not None

class TestClassifierInitializer:
    @abstractmethod
    def classifier_initializer(self) -> ClassifierInitializer:
        raise NotImplementedError("You must implement the ClassifierInitializer fixture")

    def test_running(self, qtbot):
        try:
            initializer = self.classifier_initializer()
        except NotImplementedError:
            return
        
        self.check_in_progress = False
        def in_progress():
            self.check_in_progress = True
        
        self.check_completed = False
        def completed():
            self.check_completed = True

        initializer.in_progress.connect(in_progress)
        initializer.completed.connect(completed)

        with qtbot.waitSignal(initializer.finished, timeout=300):
            initializer.start()

        assert self.check_in_progress
        assert self.check_in_progress
        assert initializer.classifier is not None


# Example usage

# class CheckClassifier(Classifier):
#     def classify_image(self):
#         self.classifier_answer = "check"

# class TestCheckClassifier(TestClassifier):
#     def classifier(self):
#         return CheckClassifier()