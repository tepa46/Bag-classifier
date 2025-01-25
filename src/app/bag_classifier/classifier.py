import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.bag_classifier.constants import datasetClasses, randomState, datasetTestPart
from app.bag_classifier.hypotheses.hypotheses import collect_image_features
from app.bag_classifier.stats.classifier_estimator import estimate_classifier
from app.bag_classifier.utils.classifier_utils import collect_images_info
from app.bag_classifier.utils.images_utils import load_image_from_filepath
from app.view.api.classifier import Classifier, ClassifierInitializer

logger = logging.getLogger(__name__)


class BagsClassifier(Classifier):
    """
    A classifier specifically designed for categorizing images of bags using a `RandomForestClassifier`.

    !!! `BagsClassifier` cannot be created directly. !!!
    !!! Use the `BagsClassifierInitializer` to get an instance of this class. !!!

    Attributes:
        classifier (`RandomForestClassifier`):
            An instance of a random forest classifier used for predictions.
            It is initialized with a fixed random state for reproducibility.
    """

    classifier: RandomForestClassifier

    def __init__(self):
        super().__init__()

        self.classifier = RandomForestClassifier(random_state=randomState)

    def classify_image(self):
        """
        Predicts the class probabilities for a single image of bag.
        """

        image = load_image_from_filepath(self.classifier_image_path)
        X = collect_image_features(image)

        result = self.classifier.predict_proba([X])

        self.classifier_answer = (
            f"{datasetClasses[0]}: {result[0][0]}\n"
            f"{datasetClasses[1]}: {result[0][1]}\n"
            f"{datasetClasses[2]}: {result[0][2]}"
        )

        logger.info(
            f"$Classifier$ Image {self.classifier_image_path} class was predicted."
            f" Prediction results: {self.classifier_answer}"
        )


class BagsClassifierInitializer(ClassifierInitializer):
    """
    Handles the initialization and training of the `BagsClassifier`.
    """

    def initialize_classifier(self):
        """
        Trains the `BagsClassifier`, and evaluates its performance.
        """

        images_info = collect_images_info()
        logger.info("$ClassifierInitializer$ image info was collected")

        X = []
        Y = []

        for image, image_class in images_info:
            X.append(collect_image_features(image))
            Y.append(image_class)

        X_train, X_test, Y_train, Y_test = train_test_split(
            np.array(X),
            np.array(Y),
            test_size=datasetTestPart,
            random_state=randomState,
        )

        self.classifier = BagsClassifier()
        self.classifier.classifier.fit(X_train, Y_train)
        logger.info("$ClassifierInitializer$ classifier has been trained")

        estimate_classifier(self.classifier, X_test, Y_test)
