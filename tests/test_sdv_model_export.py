from pathlib import Path
from unittest import mock

import mlflow
import pandas as pd
import pytest
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from pandas.testing import assert_frame_equal
from sdv.datasets.demo import download_demo
from sdv.lite import SingleTablePreset
from sdv.multi_table import HMASynthesizer

import mlflavors.sdv

NUM_ROWS = 10
SCALE = 2


@pytest.fixture
def model_path(tmp_path):
    """Create a temporary path to save/log model."""
    return tmp_path.joinpath("model")


@pytest.fixture
def sdv_custom_env(tmp_path):
    """Create a conda environment and returns path to conda environment yml file."""
    conda_env = tmp_path.joinpath("conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["sdv"])
    return conda_env


@pytest.fixture(scope="module")
def single_table():
    """Create sample data for sdv single_table model."""
    real_data, metadata = download_demo(
        modality="single_table", dataset_name="fake_hotel_guests"
    )
    return real_data, metadata


@pytest.fixture(scope="module")
def multi_table():
    """Create sample data for sdv multi_table model."""
    real_data, metadata = download_demo(
        modality="multi_table", dataset_name="fake_hotels"
    )
    return real_data, metadata


@pytest.fixture(scope="module")
def sequential_table():
    """Create sample data for sdv sequential_table model."""
    real_data, metadata = download_demo(
        modality="sequential", dataset_name="nasdaq100_2019"
    )
    return real_data, metadata


@pytest.fixture(scope="module")
def single_table_model(single_table):
    """Create instance of fitted sdv single_table_model."""
    real_data, metadata = single_table
    synthesizer = SingleTablePreset(metadata, name="FAST_ML")
    synthesizer.fit(real_data)

    return synthesizer


@pytest.fixture(scope="module")
def multi_table_model(multi_table):
    """Create instance of fitted sdv multi_table_model."""
    real_data, metadata = multi_table
    synthesizer = HMASynthesizer(metadata)
    synthesizer.fit(real_data)

    return synthesizer


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_single_table_model_save_and_load(
    single_table_model, model_path, serialization_format
):
    """Test saving and loading of native sdv single_table_model."""
    mlflavors.sdv.save_model(
        sdv_model=single_table_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_model = mlflavors.sdv.load_model(
        model_uri=model_path,
    )

    assert_frame_equal(
        single_table_model.sample(num_rows=NUM_ROWS),
        loaded_model.sample(num_rows=NUM_ROWS),
    )


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_multi_table_model_save_and_load(
    multi_table_model, model_path, serialization_format
):
    """Test saving and loading of native sdv multi_table_model."""
    mlflavors.sdv.save_model(
        sdv_model=multi_table_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_model = mlflavors.sdv.load_model(
        model_uri=model_path,
    )

    synthetic_data_multi_table_model = multi_table_model.sample(scale=SCALE)
    synthetic_data_loaded_model = loaded_model.sample(scale=SCALE)

    assert_frame_equal(
        synthetic_data_multi_table_model["hotels"],
        synthetic_data_loaded_model["hotels"],
    )

    assert_frame_equal(
        synthetic_data_multi_table_model["guests"],
        synthetic_data_loaded_model["guests"],
    )


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_single_table_model_pyfunc_output(
    single_table_model, model_path, serialization_format
):
    """Test sdv prediction of loaded pyfunc single_table_model with parameters."""
    mlflavors.sdv.save_model(
        sdv_model=single_table_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_pyfunc = mlflavors.sdv.pyfunc.load_model(model_uri=model_path)

    predict_conf = pd.DataFrame(
        [
            {
                "modality": "single_table",
                "num_rows": NUM_ROWS,
            }
        ]
    )

    model_predictions = single_table_model.sample(num_rows=NUM_ROWS)
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)

    assert_frame_equal(model_predictions, pyfunc_predict)


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_multi_table_model_pyfunc_output(
    multi_table_model, model_path, serialization_format
):
    """Test sdv prediction of loaded pyfunc multi_table_model with parameters."""
    mlflavors.sdv.save_model(
        sdv_model=multi_table_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_pyfunc = mlflavors.sdv.pyfunc.load_model(model_uri=model_path)

    predict_conf = pd.DataFrame(
        [
            {
                "modality": "multi_table",
                "scale": SCALE,
            }
        ]
    )

    model_predictions = multi_table_model.sample(scale=SCALE)
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)[0]

    assert_frame_equal(
        model_predictions["hotels"],
        pyfunc_predict["hotels"],
    )

    assert_frame_equal(
        model_predictions["guests"],
        pyfunc_predict["guests"],
    )


@pytest.mark.parametrize("use_signature", [True, False])
def test_single_table_model_signature_and_examples_saved_correctly(
    single_table_model, single_table, model_path, use_signature
):
    """Test saving of mlflow signature for native single_table_model sample method."""
    real_data, _ = single_table

    # Note: signature inference only works for 'single_table' and 'sequential'
    # modalities
    prediction = single_table_model.sample(num_rows=NUM_ROWS)
    signature = infer_signature(real_data, prediction) if use_signature else None
    mlflavors.sdv.save_model(
        single_table_model,
        path=model_path,
        signature=signature,
    )
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature


@pytest.mark.parametrize("use_signature", [True, False])
def test_single_table_model_signature_for_pyfunc_predict(
    single_table_model, model_path, use_signature, single_table
):
    """Test saving of mlflow signature for pyfunc single_table_model predict."""
    real_data, _ = single_table

    model_path_primary = model_path.joinpath("primary")
    model_path_secondary = model_path.joinpath("secondary")

    mlflavors.sdv.save_model(sdv_model=single_table_model, path=model_path_primary)
    loaded_pyfunc = mlflavors.sdv.pyfunc.load_model(model_uri=model_path_primary)

    predict_conf = pd.DataFrame(
        [
            {
                "modality": "single_table",
                "num_rows": NUM_ROWS,
            }
        ]
    )

    prediction = loaded_pyfunc.predict(predict_conf)
    signature = infer_signature(real_data, prediction) if use_signature else None
    mlflavors.sdv.save_model(
        single_table_model,
        path=model_path_secondary,
        signature=signature,
    )
    mlflow_model = Model.load(model_path_secondary)
    assert signature == mlflow_model.signature


@pytest.mark.parametrize("should_start_run", [True, False])
@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_log_model(
    single_table_model,
    tmp_path,
    should_start_run,
    serialization_format,
):
    """Test logging and reloading sdv model."""
    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "model"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["sdv"])
        model_info = mlflavors.sdv.log_model(
            sdv_model=single_table_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            serialization_format=serialization_format,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = mlflavors.sdv.load_model(
            model_uri=model_uri,
        )
        assert_frame_equal(
            single_table_model.sample(num_rows=NUM_ROWS),
            reloaded_model.sample(num_rows=NUM_ROWS),
        )
        model_path = Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
    finally:
        mlflow.end_run()


def test_log_model_calls_register_model(single_table_model, tmp_path):
    """Test log model calls register model."""
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["sdv"])
        mlflavors.sdv.log_model(
            sdv_model=single_table_model,
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


def test_log_model_no_registered_model_name(single_table_model, tmp_path):
    """Test log model calls register model without registered model name."""
    artifact_path = "sdv"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["sdv"])
        mlflavors.sdv.log_model(
            sdv_model=single_table_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        mlflow.register_model.assert_not_called()


def test_sdv_pyfunc_raises_invalid_df_input(single_table_model, model_path):
    """Test pyfunc call raises error with invalid dataframe configuration."""
    mlflavors.sdv.save_model(sdv_model=single_table_model, path=model_path)
    loaded_pyfunc = mlflavors.sdv.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(MlflowException, match="The provided prediction pd.DataFrame "):
        loaded_pyfunc.predict(
            pd.DataFrame([{"modality": "single_table"}, {"num_rows": NUM_ROWS}])
        )

    with pytest.raises(MlflowException, match="Invalid `modality` "):
        loaded_pyfunc.predict(
            pd.DataFrame([{"modality": "invalid", "num_rows": NUM_ROWS}])
        )
