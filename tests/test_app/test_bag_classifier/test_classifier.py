import pytest
from app.bag_classifier.classifier import BagsClassifierInitializer
from test_app.test_view.test_api.test_classifier import TestClassifierInitializer

class MockedRandomForestClassifier:
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        return

    def predict_proba(self, X):
        return [["first mock", "second mock", "third mock"]]
    
    def fit(self, X, y, sample_weight=None):
        return
    
@pytest.fixture(autouse=True)
def mock_random_forest_classsifier(mocker):
    mocker.patch("app.bag_classifier.classifier.RandomForestClassifier", MockedRandomForestClassifier)

@pytest.fixture(autouse=True)
def mock_utils(mocker):
    # TODO: make mocks for utils and hypos like MokedRandomForestClassifier
    # e. g. mocker.mock("app.bag_classifier.utils.say_52", mocked_say_52)
    return

class TestBagsClassifierInitializer(TestClassifierInitializer):
    @pytest.mark.usefixtures("mock_random_forest_classsifier")
    def classifier_initializer(self):
        return BagsClassifierInitializer()