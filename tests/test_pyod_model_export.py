from pathlib import Path
from unittest import mock

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from numpy.testing import assert_array_equal
from pyod.models.knn import KNN
from pyod.utils.data import generate_data

import mlflavors.pyod

SEASON_LENGTH = 12
LEVEL = [90, 95]


@pytest.fixture
def model_path(tmp_path):
    """Create a temporary path to save/log model."""
    return tmp_path.joinpath("model")


@pytest.fixture
def pyod_custom_env(tmp_path):
    """Create a conda environment and returns path to conda environment yml file."""
    conda_env = tmp_path.joinpath("conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["pyod"])
    return conda_env


@pytest.fixture(scope="module")
def data():
    """Create sample data for pyod model."""
    contamination = 0.1  # percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

    # Generate sample data
    X_train, X_test, y_train, y_test = generate_data(
        n_train=n_train,
        n_test=n_test,
        n_features=2,
        contamination=contamination,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def knn_model(data):
    """Create instance of fitted pyod model."""
    X_train, _, _, _ = data
    clf = KNN()
    return clf.fit(X_train)


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_knn_model_save_and_load(knn_model, model_path, serialization_format, data):
    """Test saving and loading of native pyod model."""
    _, X_test, _, _ = data
    mlflavors.pyod.save_model(
        pyod_model=knn_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_model = mlflavors.pyod.load_model(
        model_uri=model_path,
    )

    assert_array_equal(
        knn_model.predict(X_test, return_confidence=False),
        loaded_model.predict(X_test, return_confidence=False),
    )


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_knn_model_pyfunc_output(knn_model, model_path, serialization_format, data):
    """Test pyod prediction of loaded pyfunc model with parameters."""
    _, X_test, _, _ = data
    mlflavors.pyod.save_model(
        pyod_model=knn_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_pyfunc = mlflavors.pyod.pyfunc.load_model(model_uri=model_path)

    predict_conf = pd.DataFrame(
        [
            {
                "predict_method": "decision_function",
                "X": X_test,
            }
        ]
    )

    model_predictions = knn_model.decision_function(X_test)
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)

    assert_array_equal(model_predictions, pyfunc_predict[0])

    predict_conf = pd.DataFrame(
        [
            {
                "predict_method": "predict",
                "X": X_test,
                "return_confidence": True,
            }
        ]
    )

    y_test_pred, y_test_pred_confidence = knn_model.predict(
        X_test, return_confidence=True
    )
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)

    assert_array_equal(y_test_pred, pyfunc_predict[0][0])
    assert_array_equal(y_test_pred_confidence, pyfunc_predict[0][1])

    predict_conf = pd.DataFrame(
        [
            {
                "predict_method": "predict_proba",
                "X": X_test,
                "return_confidence": True,
                "method": "linear",
            }
        ]
    )

    y_test_pred, y_test_pred_confidence = knn_model.predict_proba(
        X_test, method="linear", return_confidence=True
    )
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)

    assert_array_equal(y_test_pred, pyfunc_predict[0][0])
    assert_array_equal(y_test_pred_confidence, pyfunc_predict[0][1])

    predict_conf = pd.DataFrame(
        [
            {
                "predict_method": "predict_confidence",
                "X": X_test,
            }
        ]
    )

    y_test_pred_confidence = knn_model.predict_confidence(X_test)
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)

    assert_array_equal(y_test_pred_confidence, pyfunc_predict[0])


@pytest.mark.parametrize("use_signature", [True, False])
@pytest.mark.parametrize("use_example", [True, False])
def test_signature_and_examples_saved_correctly(
    knn_model, data, model_path, use_signature, use_example
):
    """Test saving of mlflow signature for native pyod predict method."""
    _, X_test, _, _ = data

    prediction = knn_model.predict(X_test)
    signature = infer_signature(X_test, prediction) if use_signature else None
    example = X_test[0:5] if use_example else None
    mlflavors.pyod.save_model(
        knn_model,
        path=model_path,
        signature=signature,
        input_example=example,
    )
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature
    if example is None:
        assert mlflow_model.saved_input_example_info is None
    else:
        r_example = _read_example(mlflow_model, model_path)
        np.testing.assert_array_equal(r_example, example)


@pytest.mark.parametrize("use_signature", [True, False])
def test_signature_for_pyfunc_predict(knn_model, model_path, data, use_signature):
    """Test saving of mlflow signature for pyfunc predict."""
    _, X_test, _, _ = data

    model_path_primary = model_path.joinpath("primary")
    model_path_secondary = model_path.joinpath("secondary")

    mlflavors.pyod.save_model(pyod_model=knn_model, path=model_path_primary)
    loaded_pyfunc = mlflavors.pyod.pyfunc.load_model(model_uri=model_path_primary)

    predict_conf = pd.DataFrame(
        [
            {
                "predict_method": "predict",
                "X": X_test,
            }
        ]
    )

    forecast = loaded_pyfunc.predict(predict_conf)
    signature = infer_signature(X_test, forecast[0]) if use_signature else None
    mlflavors.pyod.save_model(
        knn_model,
        path=model_path_secondary,
        signature=signature,
    )
    mlflow_model = Model.load(model_path_secondary)
    assert signature == mlflow_model.signature


@pytest.mark.parametrize("should_start_run", [True, False])
@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_log_model(
    knn_model,
    tmp_path,
    should_start_run,
    serialization_format,
    data,
):
    """Test logging and reloading pyod model."""
    _, X_test, _, _ = data

    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "model"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["pyod"])
        model_info = mlflavors.pyod.log_model(
            pyod_model=knn_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            serialization_format=serialization_format,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = mlflavors.pyod.load_model(
            model_uri=model_uri,
        )
        np.testing.assert_array_equal(
            knn_model.predict(X_test),
            reloaded_model.predict(X_test),
        )
        model_path = Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
    finally:
        mlflow.end_run()


def test_log_model_calls_register_model(knn_model, tmp_path):
    """Test log model calls register model."""
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["pyod"])
        mlflavors.pyod.log_model(
            pyod_model=knn_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            registered_model_name="PyOD",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model.assert_called_once_with(
            model_uri,
            "PyOD",
            await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        )


def test_log_model_no_registered_model_name(knn_model, tmp_path):
    """Test log model calls register model without registered model name."""
    artifact_path = "pyod"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["pyod"])
        mlflavors.pyod.log_model(
            pyod_model=knn_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        mlflow.register_model.assert_not_called()


def test_pyod_pyfunc_raises_invalid_df_input(knn_model, model_path, data):
    """Test pyfunc call raises error with invalid dataframe configuration."""
    _, X_test, _, _ = data
    mlflavors.pyod.save_model(pyod_model=knn_model, path=model_path)
    loaded_pyfunc = mlflavors.pyod.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(MlflowException, match="The provided prediction pd.DataFrame "):
        loaded_pyfunc.predict(
            pd.DataFrame([{"X": X_test}, {"predict_method": "predict"}])
        )

    with pytest.raises(MlflowException, match="The provided prediction configuration "):
        loaded_pyfunc.predict(pd.DataFrame([{"invalid": True}]))

    with pytest.raises(MlflowException, match="Invalid `predict_method` "):
        loaded_pyfunc.predict(
            pd.DataFrame([{"X": X_test, "predict_method": "forecast"}])
        )
