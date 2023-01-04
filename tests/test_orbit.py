import pytest
from orbit.models import DLT
from orbit.utils.dataset import load_iclaims
from pandas.testing import assert_frame_equal

import mlflow_flavors


@pytest.fixture
def model_path(tmp_path):
    """Create a temporary path to save/log model."""
    return tmp_path.joinpath("model")


@pytest.fixture
def orbit_custom_env(tmp_path):
    """Create a conda environment and returns path to conda environment yml file."""
    from mlflow.utils.environment import _mlflow_conda_env

    conda_env = tmp_path.joinpath("conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["orbit"])
    return conda_env


@pytest.fixture(scope="module")
def test_data_iclaims():
    """Create sample data for orbit model."""
    return load_iclaims()


@pytest.fixture(scope="module")
def dlt_model(test_data_iclaims):
    """Create instance of fitted dlt model."""
    dlt = DLT(
        response_col="claims",
        date_col="week",
        regressor_col=["trend.unemploy", "trend.filling", "trend.job"],
        seasonality=52,
    )
    return dlt.fit(df=test_data_iclaims)


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_dlt_model_save_and_load(dlt_model, model_path, serialization_format):
    """Test saving and loading of native sktime auto_arima_model."""
    mlflow_flavors.orbit.save_model(
        sktime_model=dlt_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_model = mlflow_flavors.orbit.load_model(
        model_uri=model_path,
    )

    assert_frame_equal(
        dlt_model.predict(test_data_iclaims), loaded_model.predict(test_data_iclaims)
    )
