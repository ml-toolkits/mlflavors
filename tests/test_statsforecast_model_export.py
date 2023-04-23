from pathlib import Path
from unittest import mock

import mlflow
import numpy as np
import pandas as pd
import pytest
from datasetsforecast.m5 import M5
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from pandas.testing import assert_frame_equal
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive
from statsforecast.utils import AirPassengersDF

import mlflavors.statsforecast
from mlflavors.utils.data import load_m5

SEASON_LENGTH = 12
LEVEL = [90, 95]


@pytest.fixture
def model_path(tmp_path):
    """Create a temporary path to save/log model."""
    return tmp_path.joinpath("model")


@pytest.fixture(scope="module")
def data_path(tmpdir_factory):
    data_path = tmpdir_factory.mktemp("data")
    M5.download(data_path)
    return data_path


@pytest.fixture
def statsforecast_custom_env(tmp_path):
    """Create a conda environment and returns path to conda environment yml file."""
    conda_env = tmp_path.joinpath("conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["statsforecast"])
    return conda_env


@pytest.fixture(scope="module")
def data_air_passengers():
    """Create sample data for statsforecast model."""
    df = AirPassengersDF
    train_df = df[df.ds <= "1959-12-31"]  # 132 monthly observations for train
    test_df = df[df.ds > "1959-12-31"]  # 12 monthly observations for test
    return train_df, test_df


@pytest.fixture(scope="module")
def data_m5(data_path):
    """Create sample data for statsforecast model."""
    train_df, test_df, _ = load_m5(data_path)

    return train_df, test_df


@pytest.fixture(scope="module")
def arima_ets_model(data_air_passengers):
    """Create instance of statsforecast model."""
    train_df, _ = data_air_passengers

    models = [
        AutoARIMA(season_length=SEASON_LENGTH),
        AutoETS(season_length=SEASON_LENGTH),
        Naive(),
    ]

    sf = StatsForecast(df=train_df, models=models, freq="M", n_jobs=-1)
    return sf


@pytest.fixture(scope="module")
def arima_ets_fitted_model(data_air_passengers):
    """Create instance of fitted statsforecast model."""
    train_df, _ = data_air_passengers

    models = [
        AutoARIMA(season_length=SEASON_LENGTH),
        AutoETS(season_length=SEASON_LENGTH),
        Naive(),
    ]

    sf = StatsForecast(df=train_df, models=models, freq="M", n_jobs=-1)

    return sf.fit()


@pytest.fixture(scope="module")
def arima_with_exogenous_fitted_model(data_m5):
    """Create instance of fitted dlt model."""
    train_df, _ = data_m5

    models = [AutoARIMA(season_length=7)]

    sf = StatsForecast(df=train_df, models=models, freq="D", n_jobs=-1)

    return sf.fit()


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_arima_ets_model_save_and_load(
    arima_ets_model, model_path, serialization_format, data_air_passengers
):
    """Test saving and loading of native statsforecast model."""
    _, test_df = data_air_passengers
    mlflavors.statsforecast.save_model(
        statsforecast_model=arima_ets_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_model = mlflavors.statsforecast.load_model(
        model_uri=model_path,
    )

    horizon = len(test_df)

    assert_frame_equal(
        arima_ets_model.forecast(h=horizon), loaded_model.forecast(h=horizon)
    )


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_arima_ets_fitted_model_save_and_load(
    arima_ets_fitted_model, model_path, serialization_format, data_air_passengers
):
    """Test saving and loading of native statsforecast model."""
    _, test_df = data_air_passengers
    mlflavors.statsforecast.save_model(
        statsforecast_model=arima_ets_fitted_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_model = mlflavors.statsforecast.load_model(
        model_uri=model_path,
    )

    horizon = len(test_df)

    assert_frame_equal(
        arima_ets_fitted_model.predict(h=horizon), loaded_model.predict(h=horizon)
    )


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_arima_with_exogenous_fitted_model_save_and_load(
    arima_with_exogenous_fitted_model, model_path, serialization_format, data_m5
):
    """Test saving and loading of native statsforecast model."""
    _, test_df = data_m5
    mlflavors.statsforecast.save_model(
        statsforecast_model=arima_with_exogenous_fitted_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_model = mlflavors.statsforecast.load_model(
        model_uri=model_path,
    )

    horizon = 28

    assert_frame_equal(
        arima_with_exogenous_fitted_model.predict(h=horizon, X_df=test_df, level=LEVEL),
        loaded_model.predict(h=horizon, X_df=test_df, level=LEVEL),
    )


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_arima_ets_fitted_model_pyfunc_output(
    arima_ets_fitted_model, model_path, serialization_format, data_air_passengers
):
    """Test statsforecast prediction of loaded pyfunc model with parameters."""
    _, test_df = data_air_passengers
    mlflavors.statsforecast.save_model(
        statsforecast_model=arima_ets_fitted_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_pyfunc = mlflavors.statsforecast.pyfunc.load_model(model_uri=model_path)

    horizon = len(test_df)

    predict_conf = pd.DataFrame(
        [
            {
                "h": horizon,
                "level": LEVEL,
            }
        ]
    )

    model_predictions = arima_ets_fitted_model.predict(h=horizon, level=LEVEL)
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)

    assert_frame_equal(model_predictions, pyfunc_predict)


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_arima_with_exogenous_fitted_model_pyfunc_output(
    arima_with_exogenous_fitted_model, model_path, serialization_format, data_m5
):
    """Test statsforecast prediction of loaded pyfunc model with parameters."""
    _, test_df = data_m5
    mlflavors.statsforecast.save_model(
        statsforecast_model=arima_with_exogenous_fitted_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_pyfunc = mlflavors.statsforecast.pyfunc.load_model(model_uri=model_path)

    X_test_array = test_df.to_numpy()
    horizon = len(test_df)

    predict_conf = pd.DataFrame(
        [
            {
                "X": X_test_array,
                "X_cols": test_df.columns,
                "X_dtypes": list(test_df.dtypes),
                "h": horizon,
                "level": LEVEL,
            }
        ]
    )

    model_predictions = arima_with_exogenous_fitted_model.predict(
        h=horizon, X_df=test_df, level=LEVEL
    )
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)

    assert_frame_equal(model_predictions, pyfunc_predict)


@pytest.mark.parametrize("use_signature", [True, False])
def test_signature_and_examples_saved_correctly(
    arima_ets_fitted_model,
    data_air_passengers,
    model_path,
    use_signature,
):
    """Test saving of mlflow signature for native statsforecast predict method."""
    _, test_df = data_air_passengers

    # Note: Example inference fails due to incorrect recreation of numpy timestamp
    horizon = len(test_df)
    prediction = arima_ets_fitted_model.predict(h=horizon, level=LEVEL)
    signature = infer_signature(test_df, prediction) if use_signature else None
    mlflavors.statsforecast.save_model(
        arima_ets_fitted_model,
        path=model_path,
        signature=signature,
    )
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature


@pytest.mark.parametrize("use_signature", [True, False])
def test_signature_for_pyfunc_predict(
    arima_ets_fitted_model, model_path, data_air_passengers, use_signature
):
    """Test saving of mlflow signature for pyfunc predict."""
    _, test_df = data_air_passengers
    horizon = len(test_df)

    model_path_primary = model_path.joinpath("primary")
    model_path_secondary = model_path.joinpath("secondary")

    mlflavors.statsforecast.save_model(
        statsforecast_model=arima_ets_fitted_model, path=model_path_primary
    )
    loaded_pyfunc = mlflavors.statsforecast.pyfunc.load_model(
        model_uri=model_path_primary
    )

    predict_conf = pd.DataFrame(
        [
            {
                "h": horizon,
                "level": LEVEL,
            }
        ]
    )

    forecast = loaded_pyfunc.predict(predict_conf)
    signature = infer_signature(test_df, forecast) if use_signature else None
    mlflavors.statsforecast.save_model(
        arima_ets_fitted_model,
        path=model_path_secondary,
        signature=signature,
    )
    mlflow_model = Model.load(model_path_secondary)
    assert signature == mlflow_model.signature


@pytest.mark.parametrize("should_start_run", [True, False])
@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_log_model(
    arima_ets_fitted_model,
    tmp_path,
    should_start_run,
    serialization_format,
    data_air_passengers,
):
    """Test logging and reloading statsforecast model."""
    _, test_df = data_air_passengers
    horizon = len(test_df)

    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "model"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["statsforecast"])
        model_info = mlflavors.statsforecast.log_model(
            statsforecast_model=arima_ets_fitted_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            serialization_format=serialization_format,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = mlflavors.statsforecast.load_model(
            model_uri=model_uri,
        )
        np.testing.assert_array_equal(
            arima_ets_fitted_model.predict(h=horizon, level=LEVEL),
            reloaded_model.predict(h=horizon, level=LEVEL),
        )
        model_path = Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
    finally:
        mlflow.end_run()


def test_log_model_calls_register_model(arima_ets_fitted_model, tmp_path):
    """Test log model calls register model."""
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["statsforecast"])
        mlflavors.statsforecast.log_model(
            statsforecast_model=arima_ets_fitted_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            registered_model_name="StatsforecastModel",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model.assert_called_once_with(
            model_uri,
            "StatsforecastModel",
            await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        )


def test_log_model_no_registered_model_name(arima_ets_fitted_model, tmp_path):
    """Test log model calls register model without registered model name."""
    artifact_path = "statsforecast"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["statsforecast"])
        mlflavors.statsforecast.log_model(
            statsforecast_model=arima_ets_fitted_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        mlflow.register_model.assert_not_called()


def test_statsforecast_pyfunc_raises_invalid_df_input(
    arima_ets_fitted_model, model_path
):
    """Test pyfunc call raises error with invalid dataframe configuration."""
    mlflavors.statsforecast.save_model(
        statsforecast_model=arima_ets_fitted_model, path=model_path
    )
    loaded_pyfunc = mlflavors.statsforecast.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(MlflowException, match="The provided prediction pd.DataFrame "):
        loaded_pyfunc.predict(pd.DataFrame([{"h": 1}, {"level": LEVEL}]))

    with pytest.raises(MlflowException, match="The provided prediction configuration "):
        loaded_pyfunc.predict(pd.DataFrame([{"invalid": True}]))


def test_statsforecast_save_model_raises_invalid_serialization_format(
    arima_ets_fitted_model, model_path
):
    """Test save_model call raises error with invalid serialization format."""
    with pytest.raises(MlflowException, match="Unrecognized serialization format: "):
        mlflavors.statsforecast.save_model(
            statsforecast_model=arima_ets_fitted_model,
            path=model_path,
            serialization_format="json",
        )
