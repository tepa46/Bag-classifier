import logging

from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def estimate_classifier(classifier, X_test, Y_test):
    """
    Estimates and logs the performance of the given classifier on the test data.
    """

    Y_pred = classifier.classifier.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    logger.info(f"$Classifier estimation$ Accuracy: {accuracy:.4f}")
