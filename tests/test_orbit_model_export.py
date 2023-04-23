from pathlib import Path
from unittest import mock

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from orbit.models import DLT
from orbit.utils.dataset import load_iclaims
from pandas.testing import assert_frame_equal

import mlflavors.orbit

SEED = 2023
DECOMPOSE = True
STORE_PREDICTION_ARRAY = True


@pytest.fixture
def model_path(tmp_path):
    """Create a temporary path to save/log model."""
    return tmp_path.joinpath("model")


@pytest.fixture
def orbit_custom_env(tmp_path):
    """Create a conda environment and returns path to conda environment yml file."""
    conda_env = tmp_path.joinpath("conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["orbit"])
    return conda_env


@pytest.fixture(scope="module")
def data_iclaims():
    """Create sample data for orbit model."""
    df = load_iclaims()
    test_size = 52
    train_df = df[:-test_size]
    test_df = df[-test_size:]
    return train_df, test_df


@pytest.fixture(scope="module")
def dlt_model(data_iclaims):
    """Create instance of fitted dlt model."""
    train_df, _ = data_iclaims
    dlt = DLT(
        response_col="claims",
        date_col="week",
        regressor_col=["trend.unemploy", "trend.filling", "trend.job"],
        seasonality=52,
    )
    return dlt.fit(df=train_df)


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_dlt_model_save_and_load(
    dlt_model, model_path, serialization_format, data_iclaims
):
    """Test saving and loading of native orbit dlt model."""
    _, test_df = data_iclaims
    mlflavors.orbit.save_model(
        orbit_model=dlt_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_model = mlflavors.orbit.load_model(
        model_uri=model_path,
    )

    assert_frame_equal(
        dlt_model.predict(test_df, seed=SEED), loaded_model.predict(test_df, seed=SEED)
    )


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_dlt_model_pyfunc_output(
    dlt_model, model_path, serialization_format, data_iclaims
):
    """Test dlt prediction of loaded pyfunc model with parameters."""
    _, test_df = data_iclaims
    mlflavors.orbit.save_model(
        orbit_model=dlt_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_pyfunc = mlflavors.orbit.pyfunc.load_model(model_uri=model_path)

    X_test_array = test_df.to_numpy()
    predict_conf = pd.DataFrame(
        [
            {
                "X": X_test_array,
                "X_cols": test_df.columns,
                "X_dtypes": list(test_df.dtypes),
                "decompose": DECOMPOSE,
                "store_prediction_array": STORE_PREDICTION_ARRAY,
                "seed": SEED,
            }
        ]
    )

    model_predictions = dlt_model.predict(
        test_df,
        decompose=DECOMPOSE,
        store_prediction_array=STORE_PREDICTION_ARRAY,
        seed=SEED,
    )
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)

    assert_frame_equal(model_predictions, pyfunc_predict)


@pytest.mark.parametrize("use_signature", [True, False])
def test_signature_and_examples_saved_correctly(
    dlt_model,
    data_iclaims,
    model_path,
    use_signature,
):
    """Test saving of mlflow signature for native orbit predict method."""
    _, test_df = data_iclaims

    # Note: Example inference fails due to incorrect recreation of numpy timestamp
    prediction = dlt_model.predict(test_df)
    signature = infer_signature(test_df, prediction) if use_signature else None
    mlflavors.orbit.save_model(
        dlt_model,
        path=model_path,
        signature=signature,
    )
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature


@pytest.mark.parametrize("use_signature", [True, False])
def test_signature_for_pyfunc_predict(
    dlt_model, model_path, data_iclaims, use_signature
):
    """Test saving of mlflow signature for pyfunc predict."""
    _, test_df = data_iclaims

    model_path_primary = model_path.joinpath("primary")
    model_path_secondary = model_path.joinpath("secondary")

    mlflavors.orbit.save_model(orbit_model=dlt_model, path=model_path_primary)
    loaded_pyfunc = mlflavors.orbit.pyfunc.load_model(model_uri=model_path_primary)

    X_test_array = test_df.to_numpy()
    predict_conf = pd.DataFrame(
        [
            {
                "X": X_test_array,
                "X_cols": test_df.columns,
                "X_dtypes": list(test_df.dtypes),
                "decompose": DECOMPOSE,
                "store_prediction_array": STORE_PREDICTION_ARRAY,
                "seed": SEED,
            }
        ]
    )

    forecast = loaded_pyfunc.predict(predict_conf)
    signature = infer_signature(test_df, forecast) if use_signature else None
    mlflavors.orbit.save_model(
        dlt_model,
        path=model_path_secondary,
        signature=signature,
    )
    mlflow_model = Model.load(model_path_secondary)
    assert signature == mlflow_model.signature


@pytest.mark.parametrize("should_start_run", [True, False])
@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_log_model(
    dlt_model, tmp_path, should_start_run, serialization_format, data_iclaims
):
    """Test logging and reloading orbit model."""
    _, test_df = data_iclaims

    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "model"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["orbit"])
        model_info = mlflavors.orbit.log_model(
            orbit_model=dlt_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            serialization_format=serialization_format,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = mlflavors.orbit.load_model(
            model_uri=model_uri,
        )
        np.testing.assert_array_equal(
            dlt_model.predict(test_df, seed=SEED),
            reloaded_model.predict(test_df, seed=SEED),
        )
        model_path = Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
    finally:
        mlflow.end_run()


def test_log_model_calls_register_model(dlt_model, tmp_path):
    """Test log model calls register model."""
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["orbit"])
        mlflavors.orbit.log_model(
            orbit_model=dlt_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            registered_model_name="OrbitModel",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model.assert_called_once_with(
            model_uri,
            "OrbitModel",
            await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        )


def test_log_model_no_registered_model_name(dlt_model, tmp_path):
    """Test log model calls register model without registered model name."""
    artifact_path = "orbit"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["orbit"])
        mlflavors.orbit.log_model(
            orbit_model=dlt_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        mlflow.register_model.assert_not_called()


def test_orbit_pyfunc_raises_invalid_df_input(dlt_model, model_path):
    """Test pyfunc call raises error with invalid dataframe configuration."""
    mlflavors.orbit.save_model(orbit_model=dlt_model, path=model_path)
    loaded_pyfunc = mlflavors.orbit.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(MlflowException, match="The provided prediction pd.DataFrame "):
        loaded_pyfunc.predict(pd.DataFrame([{"decompose": DECOMPOSE}, {"seed": SEED}]))

    with pytest.raises(MlflowException, match="The provided prediction configuration "):
        loaded_pyfunc.predict(pd.DataFrame([{"invalid": True}]))


def test_orbit_save_model_raises_invalid_serialization_format(dlt_model, model_path):
    """Test save_model call raises error with invalid serialization format."""
    with pytest.raises(MlflowException, match="Unrecognized serialization format: "):
        mlflavors.orbit.save_model(
            orbit_model=dlt_model, path=model_path, serialization_format="json"
        )
