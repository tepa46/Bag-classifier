from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog

from app.view.api.classifier import Classifier, ClassifierInitializer
from app.view.constants import ApplicationName, WindowHeight, WindowWidth, LoadingWindowLabel, ClassifyImageLabel, \
    ClassificationActionLabel


class ImageClassifierApp(QWidget):
    """
        Main application window for the Image Classifier.

        Attributes:
            _loading_widget (QLabel): A widget displaying a loading message during classifier initialization.
            _classify_button (QPushButton): A button to trigger the image classification action.
            _image_widget (QLabel): A widget to display the selected image.
            _classification_action_widget (QLabel): A widget showing the classification action message.
            _classification_result_widget (QLabel): A widget displaying the classification results.
            _classifier_initializer (ClassifierInitializer): The instance responsible for initializing the classifier.
            _classifier (Classifier): The classifier used to classify images.
        """

    _loading_widget: QLabel
    _classify_button: QPushButton
    _image_widget: QLabel
    _classification_action_widget: QLabel
    _classification_result_widget: QLabel

    _classifier_initializer: ClassifierInitializer
    _classifier: Classifier

    def __init__(self, classifier_initializer: ClassifierInitializer):
        """
        Initializes the BagClassifierApp window and its components.

        Parameters:
            classifier_initializer (ClassifierInitializer): The initializer responsible for setting up the classifier.
        """

        super().__init__()

        self.setWindowTitle(ApplicationName)

        self.setFixedHeight(WindowHeight)
        self.setFixedWidth(WindowWidth)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self._loading_widget = None
        self._classify_button = None
        self._image_widget = None
        self._classification_action_widget = None
        self._classification_result_widget = None

        self._classifier_initializer = classifier_initializer
        self._classifier = None

        self.init_ui()

    @property
    def loading_widget(self):
        """
        Provides the loading widget that is shown during classifier initialization.
        """

        if not self._loading_widget:
            self._loading_widget = QLabel(LoadingWindowLabel, self)
        return self._loading_widget

    @loading_widget.deleter
    def loading_widget(self):
        """
        Deletes the loading widget from layout.
        """

        self.loading_widget.deleteLater()
        self._loading_widget = None

    @property
    def classification_action_widget(self):
        """
        Provides the classification action widget shown while classification is in progress.
        """

        if not self._classification_action_widget:
            self._classification_action_widget = QLabel(ClassificationActionLabel, self)
        return self._classification_action_widget

    @classification_action_widget.deleter
    def classification_action_widget(self):
        """
        Deletes the classification action widget from layout.
        """

        self.classification_action_widget.deleteLater()
        self._classification_action_widget = None

    @property
    def classification_result_widget(self):
        """
        Provides the widget displaying the classification result after processing the image.
        """

        if not self._classification_result_widget:
            result = self._classifier.classifier_answer
            self._classification_result_widget = QLabel(result, self)
        return self._classification_result_widget

    @classification_result_widget.deleter
    def classification_result_widget(self):
        """
        Deletes the classification result widget from layout.
        """

        self.classification_result_widget.deleteLater()
        self._classification_result_widget = None

    @property
    def classify_button(self):
        """
        Provides the classify button that triggers the image classification action.
        """

        if not self._classify_button:
            self._classify_button = QPushButton(ClassifyImageLabel, self)
            self._classify_button.clicked.connect(self.classify_image_action)

        return self._classify_button

    @property
    def image_widget(self):
        """
        Provides the widget that displays the selected image.
        """

        if not self._image_widget:
            self._image_widget = QLabel(self)
        return self._image_widget

    @image_widget.setter
    def image_widget(self, pixmap):
        """
        Sets the pixmap of the image widget.

        Parameters:
            pixmap (QPixmap): The image to display in the widget.
        """

        self.image_widget.setPixmap(pixmap)

    def init_ui(self):
        """
        Start classifier initialization with showing loading screen.
        """
        self._classifier_initializer.in_progress.connect(self.show_loading_screen)
        self._classifier_initializer.completed.connect(self.show_choosing_screen)
        self._classifier_initializer.start()

    def show_loading_screen(self):
        """
        Displays the loading screen while the classifier is being initialized.
        """

        self.layout.addWidget(self.loading_widget, alignment=Qt.AlignCenter)

    def show_choosing_screen(self):
        """
        Displays the image choosing screen.
        """

        del self.loading_widget

        self._classifier = self._classifier_initializer.classifier

        self.layout.addWidget(self.classify_button, alignment=Qt.AlignTop)
        self.layout.addWidget(self.image_widget, alignment=Qt.AlignCenter)

    def show_classification_action_screen(self):
        """
        Displays the classification action screen during the classification process.
        """

        del self.classification_result_widget

        self.layout.addWidget(self.classification_action_widget, alignment=Qt.AlignCenter)

    def show_classification_results_screen(self):
        """
        Displays the classification result screen once the classification is completed.
        """

        del self.classification_action_widget

        self.layout.addWidget(self.classification_result_widget, alignment=Qt.AlignCenter)

    def classify_image_action(self):
        """
        Opens a file dialog for the user to select an image file for classification.

        Once an image is selected, it updates the image widget and starts the classification process.
        """

        file_path, _ = QFileDialog.getOpenFileName(self, 'Choose image', '', 'JPEG Files (*.jpg *.jpeg)')

        if file_path:
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(300, 300, aspectRatioMode=True)

            self.image_widget = pixmap

            self._classifier.set_image_to_classify(file_path)
            self._classifier.in_progress.connect(self.show_classification_action_screen)
            self._classifier.completed.connect(self.show_classification_results_screen)
            self._classifier.start()
