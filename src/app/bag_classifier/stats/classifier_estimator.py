import logging

from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


def estimate_classifier(classifier, X_test, Y_test):
    """
    Estimates and logs the performance of the given classifier on the test data.
    """

    Y_pred = classifier.classifier.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    logger.info(f"$Classifier estimation$ Accuracy: {accuracy:.4f}")

    f1 = f1_score(Y_test, Y_pred, average="weighted")
    print(f"Weighted F1-score: {f1:.2f}")

    f1_per_class = f1_score(Y_test, Y_pred, average=None)
    print(f"F1-score per class: {f1_per_class}")
