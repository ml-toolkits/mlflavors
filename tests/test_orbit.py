from pathlib import Path
from unittest import mock

import mlflow
import numpy as np
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

import mlflow_flavors
import mlflow_flavors.orbit
from mlflow_flavors.orbit import PYFUNC_PREDICT_CONF


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
    df = load_iclaims()
    test_size = 52
    train_df = df[:-test_size]
    test_df = df[-test_size:]
    return train_df, test_df


@pytest.fixture(scope="module")
def dlt_model(test_data_iclaims):
    """Create instance of fitted dlt model."""
    train_df, _ = test_data_iclaims
    dlt = DLT(
        response_col="claims",
        date_col="week",
        regressor_col=["trend.unemploy", "trend.filling", "trend.job"],
        seasonality=52,
    )
    return dlt.fit(df=train_df)


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_dlt_model_save_and_load(
    dlt_model, model_path, serialization_format, test_data_iclaims
):
    """Test saving and loading of native orbit dlt model."""
    _, test_df = test_data_iclaims
    mlflow_flavors.orbit.save_model(
        orbit_model=dlt_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_model = mlflow_flavors.orbit.load_model(
        model_uri=model_path,
    )

    assert_frame_equal(
        dlt_model.predict(test_df, seed=43), loaded_model.predict(test_df, seed=43)
    )


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_dlt_model_pyfunc_without_params_output(
    dlt_model, model_path, serialization_format, test_data_iclaims
):
    """Test dlt prediction of loaded pyfunc model without parameters."""
    delattr(dlt_model, PYFUNC_PREDICT_CONF) if hasattr(
        dlt_model, PYFUNC_PREDICT_CONF
    ) else None

    _, test_df = test_data_iclaims
    mlflow_flavors.orbit.save_model(
        orbit_model=dlt_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_pyfunc = mlflow_flavors.orbit.pyfunc.load_model(model_uri=model_path)

    model_predictions = dlt_model.predict(test_df)
    pyfunc_predict = loaded_pyfunc.predict(test_df)

    assert len(model_predictions) == len(pyfunc_predict)


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_dlt_model_pyfunc_with_params_output(
    dlt_model, model_path, serialization_format, test_data_iclaims
):
    """Test dlt prediction of loaded pyfunc model with parameters."""
    _, test_df = test_data_iclaims

    dlt_model.pyfunc_predict_conf = {
        "decompose": True,
        "store_prediction_array": True,
        "seed": 43,
    }
    mlflow_flavors.orbit.save_model(
        orbit_model=dlt_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_pyfunc = mlflow_flavors.orbit.pyfunc.load_model(model_uri=model_path)

    model_predictions = dlt_model.predict(
        test_df, decompose=True, store_prediction_array=True, seed=43
    )
    pyfunc_predict = loaded_pyfunc.predict(test_df)

    assert_frame_equal(model_predictions, pyfunc_predict)


@pytest.mark.parametrize("use_signature", [True, False])
def test_signature_and_examples_saved_correctly(
    dlt_model,
    test_data_iclaims,
    model_path,
    use_signature,
):
    """Test saving of mlflow signature for native orbit predict method."""
    _, test_df = test_data_iclaims

    # Note: Example inference fails due to incorrect recreation of numpy timestamp
    prediction = dlt_model.predict(test_df)
    signature = infer_signature(test_df, prediction) if use_signature else None
    mlflow_flavors.orbit.save_model(
        dlt_model,
        path=model_path,
        signature=signature,
    )
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature


@pytest.mark.parametrize("use_signature", [True, False])
def test_signature_for_pyfunc_predict(
    dlt_model, model_path, test_data_iclaims, use_signature
):
    """Test saving of mlflow signature for pyfunc predict."""
    _, test_df = test_data_iclaims

    model_path_primary = model_path.joinpath("primary")
    model_path_secondary = model_path.joinpath("secondary")
    dlt_model.pyfunc_predict_conf = {
        "seed": 43,
    }
    mlflow_flavors.orbit.save_model(orbit_model=dlt_model, path=model_path_primary)
    loaded_pyfunc = mlflow_flavors.orbit.pyfunc.load_model(model_uri=model_path_primary)

    forecast = loaded_pyfunc.predict(test_df)
    signature = infer_signature(test_df, forecast) if use_signature else None
    mlflow_flavors.orbit.save_model(
        dlt_model,
        path=model_path_secondary,
        signature=signature,
    )
    mlflow_model = Model.load(model_path_secondary)
    assert signature == mlflow_model.signature


@pytest.mark.parametrize("should_start_run", [True, False])
@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_log_model(
    dlt_model, tmp_path, should_start_run, serialization_format, test_data_iclaims
):
    """Test logging and reloading orbit model."""
    _, test_df = test_data_iclaims

    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "model"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["orbit"])
        model_info = mlflow_flavors.orbit.log_model(
            orbit_model=dlt_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            serialization_format=serialization_format,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = mlflow_flavors.orbit.load_model(
            model_uri=model_uri,
        )
        np.testing.assert_array_equal(
            dlt_model.predict(test_df, seed=43),
            reloaded_model.predict(test_df, seed=43),
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
        mlflow_flavors.orbit.log_model(
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


def test_pyfunc_raises_invalid_attribute_type(dlt_model, model_path, test_data_iclaims):
    """Test pyfunc raises exception with invalid attribute type."""
    _, test_df = test_data_iclaims

    dlt_model.pyfunc_predict_conf = ("seed", 43)
    mlflow_flavors.orbit.save_model(orbit_model=dlt_model, path=model_path)
    loaded_pyfunc = mlflow_flavors.orbit.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(
        MlflowException,
        match=f"Attribute {PYFUNC_PREDICT_CONF} must be of type dict.",
    ):
        loaded_pyfunc.predict(test_df)


def test_pyfunc_raises_invalid_decompose_type(dlt_model, model_path, test_data_iclaims):
    """Test pyfunc raises exception with invalid decompose type."""
    _, test_df = test_data_iclaims

    dlt_model.pyfunc_predict_conf = {
        "decompose": None,
    }
    mlflow_flavors.orbit.save_model(orbit_model=dlt_model, path=model_path)
    loaded_pyfunc = mlflow_flavors.orbit.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(
        MlflowException,
        match="The provided `decompose` value ",
    ):
        loaded_pyfunc.predict(test_df)


def test_pyfunc_raises_invalid_store_prediction_array_type(
    dlt_model, model_path, test_data_iclaims
):
    """Test pyfunc raises exception with invalid store_prediction_array type."""
    _, test_df = test_data_iclaims

    dlt_model.pyfunc_predict_conf = {
        "store_prediction_array": None,
    }
    mlflow_flavors.orbit.save_model(orbit_model=dlt_model, path=model_path)
    loaded_pyfunc = mlflow_flavors.orbit.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(
        MlflowException,
        match="The provided `store_prediction_array` value ",
    ):
        loaded_pyfunc.predict(test_df)


def test_pyfunc_raises_invalid_seed_type(dlt_model, model_path, test_data_iclaims):
    """Test pyfunc raises exception with invalid store_prediction_array type."""
    _, test_df = test_data_iclaims

    dlt_model.pyfunc_predict_conf = {
        "seed": "43",
    }
    mlflow_flavors.orbit.save_model(orbit_model=dlt_model, path=model_path)
    loaded_pyfunc = mlflow_flavors.orbit.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(
        MlflowException,
        match="The provided `seed` value ",
    ):
        loaded_pyfunc.predict(test_df)
