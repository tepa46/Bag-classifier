import pytest
from pathlib import Path
import shutil
from app.loader.dataset_loader import is_dataset_loaded, load_dataset

mocked_datasetKagglePath = "moced_datasetKagglePath"
mocked_datasetSavePath = str(Path(Path.cwd(), "mocked_datasetSavePath"))


@pytest.fixture(autouse=True)
def mock_constants(mocker):
    mocker.patch(
        "app.loader.dataset_loader.datasetKagglePath", mocked_datasetKagglePath
    )
    mocker.patch("app.loader.dataset_loader.datasetSavePath", mocked_datasetSavePath)


def remove_mocked_dataset_dir():
    dir_path = Path(mocked_datasetSavePath)
    if dir_path.exists and dir_path.is_dir():
        shutil.rmtree(dir_path)


@pytest.fixture()
def no_dataset_env():
    remove_mocked_dataset_dir()


@pytest.mark.usefixtures("no_dataset_env")
def test_is_dataset_loaded_no_dataset():
    assert not is_dataset_loaded()


@pytest.fixture()
def with_dataset_env(request):
    Path.mkdir(mocked_datasetSavePath)
    request.addfinalizer(remove_mocked_dataset_dir)


@pytest.mark.usefixtures("with_dataset_env")
def test_is_dataset_loaded_dataset_dir_exists():
    assert is_dataset_loaded()


class MockedKaggleApi:
    def authenticate(self):
        pass

    def dataset_download_files(
        self, dataset, path=None, force=False, quiet=True, unzip=False, licenses=[]
    ):
        assert dataset == mocked_datasetKagglePath
        Path(path, "tmp.txt").touch(exist_ok=True)


@pytest.fixture()
def mock_kaggle_api(mocker, request):
    mocker.patch("app.loader.dataset_loader.KaggleApi", MockedKaggleApi)
    request.addfinalizer(remove_mocked_dataset_dir)


@pytest.mark.usefixtures("mock_kaggle_api")
def test_load_dataset():
    load_dataset()
    assert is_dataset_loaded()
