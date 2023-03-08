"""The ``mlflow_flavors.orbit`` module provides an MLflow API for ``orbit`` models.

This module exports ``orbit`` models in the following formats:

orbit (native) format
    This is the main flavor that can be loaded back into orbit, which relies on pickle
    internally to serialize a model.
mlflow.pyfunc
    Produced for use by generic pyfunc-based deployment tools and batch inference.

    The `pyfunc` flavor of the model supports orbit `predict()` method.

    The interface for utilizing orbit models loaded as a `pyfunc` type for
    generating forecasts requires passing an exogenous regressor as Pandas
    DataFrame to the `pyfunc.predict()` method. The configuration of parameter
    values passed to the predict method is defined by a dictionary to be saved
    as an attribute of the fitted orbit model instance (for example
    `{"decompose": True, "store_prediction_array": True, seed = 43}`). If no
    prediction configuration is defined `pyfunc.predict()` will return output
    from orbit `predict()` method using default parameter values.
"""

import logging
import os
import pickle

import mlflow
import orbit
import pandas as pd
import yaml
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

import mlflow_flavors

FLAVOR_NAME = "orbit"

SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"
SUPPORTED_SERIALIZATION_FORMATS = [
    SERIALIZATION_FORMAT_PICKLE,
    SERIALIZATION_FORMAT_CLOUDPICKLE,
]

_logger = logging.getLogger(__name__)


def get_default_pip_requirements(include_cloudpickle=False):
    """Create list of default pip requirements for MLflow Models.

    Returns
    -------
    list of default pip requirements for MLflow Models produced by this flavor.
    Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
    that, at a minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("orbit")]
    if include_cloudpickle:
        pip_deps += [_get_pinned_requirement("cloudpickle")]

    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """Return default Conda environment for MLflow Models.

    Returns
    -------
    The default Conda environment for MLflow Models produced by calls to
    :func:`save_model()` and :func:`log_model()`
    """
    return _mlflow_conda_env(
        additional_pip_deps=get_default_pip_requirements(include_cloudpickle)
    )


def save_model(
    orbit_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature=None,
    input_example=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    serialization_format=SERIALIZATION_FORMAT_PICKLE,
):
    """Save a orbit model to a path on the local file system.

    Parameters
    ----------
    orbit_model :
        Fitted orbit model object.
    path : str
        Local path where the model is to be saved.
    conda_env : Union[dict, str], optional (default=None)
        Either a dictionary representation of a Conda environment or the path to a
        conda environment yaml file.
    code_paths : array-like, optional (default=None)
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are *prepended* to the system path
        when the model is loaded.
    mlflow_model: mlflow.models.Model, optional (default=None)
        mlflow.models.Model configuration to which to add the python_function flavor.
    signature : mlflow.models.signature.ModelSignature, optional (default=None)
        Model Signature mlflow.models.ModelSignature describes
        model input and output :py:class:`Schema <mlflow.types.Schema>`. The model
        signature can be :py:func:`inferred <mlflow.models.infer_signature>` from
        datasets with valid model input (e.g. the training dataset with target column
        omitted) and valid model output (e.g. model predictions generated on the
        training dataset), for example:

        .. code-block:: py

          from mlflow.models.signature import infer_signature

          train = df.drop_column("target_label")
          predictions = ...  # compute model predictions
          signature = infer_signature(train, predictions)
    input_example : Union[pandas.core.frame.DataFrame, numpy.ndarray, dict, list, csr_matrix, csc_matrix], optional (default=None)
        Input example provides one or several instances of valid model input.
        The example can be used as a hint of what data to feed the model. The given
        example will be converted to a ``Pandas DataFrame`` and then serialized to json
        using the ``Pandas`` split-oriented format. Bytes are base64-encoded.
    pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["orbit", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    extra_pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["pandas", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    serialization_format : str, optional (default="pickle")
        The format in which to serialize the model. This should be one of the formats
        "pickle" or "cloudpickle"
    """  # noqa: E501
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                "Unrecognized serialization format: {serialization_format}. "
                "Please specify one of the following supported formats: "
                "{supported_formats}.".format(
                    serialization_format=serialization_format,
                    supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
                )
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_subpath = "model.pkl"
    model_data_path = os.path.join(path, model_data_subpath)
    _save_model(orbit_model, model_data_path, serialization_format=serialization_format)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow_flavors.orbit",
        model_path=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        orbit_version=orbit.__version__,
        serialization_format=serialization_format,
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            include_cloudpickle = (
                serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE
            )
            default_reqs = get_default_pip_requirements(include_cloudpickle)
            # To ensure `_load_pyfunc` can successfully load the model during the
            # dependency inference, `mlflow_model.save` must be called beforehand
            # to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                model_data_path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    orbit_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature=None,
    input_example=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    serialization_format=SERIALIZATION_FORMAT_PICKLE,
    **kwargs,
):
    """
    Log a orbit model as an MLflow artifact for the current run.

    Parameters
    ----------
    orbit_model : fitted orbit model
        Fitted orbit model object.
    artifact_path : str
        Run-relative artifact path to save the model to.
    conda_env : Union[dict, str], optional (default=None)
        Either a dictionary representation of a Conda environment or the path to a
        conda environment yaml file.
    code_paths : array-like, optional (default=None)
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are *prepended* to the system path
        when the model is loaded.
    registered_model_name : str, optional (default=None)
        If given, create a model version under ``registered_model_name``, also creating
        a registered model if one with the given name does not exist.
    signature : mlflow.models.signature.ModelSignature, optional (default=None)
        Model Signature mlflow.models.ModelSignature describes
        model input and output :py:class:`Schema <mlflow.types.Schema>`. The model
        signature can be :py:func:`inferred <mlflow.models.infer_signature>` from
        datasets with valid model input (e.g. the training dataset with target column
        omitted) and valid model output (e.g. model predictions generated on the
        training dataset), for example:

        .. code-block:: py

          from mlflow.models.signature import infer_signature

          train = df.drop_column("target_label")
          predictions = ...  # compute model predictions
          signature = infer_signature(train, predictions)
    input_example : Union[pandas.core.frame.DataFrame, numpy.ndarray, dict, list, csr_matrix, csc_matrix], optional (default=None)
        Input example provides one or several instances of valid model input.
        The example can be used as a hint of what data to feed the model. The given
        example will be converted to a ``Pandas DataFrame`` and then serialized to json
        using the ``Pandas`` split-oriented format. Bytes are base64-encoded.
    await_registration_for : int, optional (default=None)
        Number of seconds to wait for the model version to finish being created and is
        in ``READY`` status. By default, the function waits for five minutes. Specify 0
        or None to skip waiting.
    pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["orbit", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    extra_pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["pandas", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    serialization_format : str, optional (default="pickle")
        The format in which to serialize the model. This should be one of the formats
        "pickle" or "cloudpickle"
    kwargs:
        Additional arguments for :py:class:`mlflow.models.model.Model`

    Returns
    -------
    A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
    metadata of the logged model.
    """  # noqa: E501
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow_flavors.orbit,
        registered_model_name=registered_model_name,
        orbit_model=orbit_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        serialization_format=serialization_format,
        **kwargs,
    )


def load_model(model_uri, dst_path=None):
    """
    Load a orbit model from a local file or a run.

    Parameters
    ----------
    model_uri : str
        The location, in URI format, of the MLflow model. For example:

                    - ``/Users/me/path/to/local/model``
                    - ``relative/path/to/local/model``
                    - ``s3://my_bucket/path/to/model``
                    - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                    - ``mlflow-artifacts:/path/to/model``

        For more information about supported URI schemes, see
        `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
        artifact-locations>`_.
    dst_path : str, optional (default=None)
        The local filesystem path to which to download the model artifact.This
        directory must already exist. If unspecified, a local output path will
        be created.

    Returns
    -------
    A orbit model instance.
    """  # noqa: E501
    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path
    )
    flavor_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME
    )
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    orbit_model_file_path = os.path.join(local_model_path, flavor_conf["pickled_model"])
    serialization_format = flavor_conf.get(
        "serialization_format", SERIALIZATION_FORMAT_PICKLE
    )
    return _load_model(
        path=orbit_model_file_path, serialization_format=serialization_format
    )


def _save_model(model, path, serialization_format):
    with open(path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            pickle.dump(model, out)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            cloudpickle.dump(model, out)
        else:
            raise MlflowException(
                message="Unrecognized serialization format: "
                "{serialization_format}".format(
                    serialization_format=serialization_format
                ),
                error_code=INTERNAL_ERROR,
            )


def _load_model(path, serialization_format):
    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                "Unrecognized serialization format: {serialization_format}. "
                "Please specify one of the following supported formats: "
                "{supported_formats}.".format(
                    serialization_format=serialization_format,
                    supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
                )
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    with open(path, "rb") as pickled_model:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            return pickle.load(pickled_model)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            return cloudpickle.load(pickled_model)


def _load_pyfunc(path):
    """Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Parameters
    ----------
    path : str
        Local filesystem path to the MLflow Model with the orbit flavor.
    """  # noqa: E501
    if os.path.isfile(path):
        serialization_format = SERIALIZATION_FORMAT_PICKLE
        _logger.warning(
            "Loading procedure in older versions of MLflow using pickle.load()"
        )
    else:
        try:
            orbit_flavor_conf = _get_flavor_configuration(
                model_path=path, flavor_name=FLAVOR_NAME
            )
            serialization_format = orbit_flavor_conf.get(
                "serialization_format", SERIALIZATION_FORMAT_PICKLE
            )
        except MlflowException:
            _logger.warning(
                "Could not find orbit flavor configuration during model "
                "loading process. Assuming 'pickle' serialization format."
            )
            serialization_format = SERIALIZATION_FORMAT_PICKLE

        pyfunc_flavor_conf = _get_flavor_configuration(
            model_path=path, flavor_name=pyfunc.FLAVOR_NAME
        )
        path = os.path.join(path, pyfunc_flavor_conf["model_path"])

    return _OrbitModelWrapper(
        _load_model(path, serialization_format=serialization_format)
    )


class _OrbitModelWrapper:
    def __init__(self, orbit_model):
        self.orbit_model = orbit_model

    def predict(self, dataframe) -> pd.DataFrame:
        df_schema = dataframe.columns.values.tolist()

        if len(dataframe) > 1:
            raise MlflowException(
                f"The provided prediction pd.DataFrame contains {len(dataframe)} rows. "
                "Only 1 row should be supplied.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        attrs = dataframe.to_dict(orient="index").get(0)
        X = attrs.get("X")
        X_cols = attrs.get("X_cols")
        decompose = attrs.get("decompose", False)
        store_prediction_array = attrs.get("store_prediction_array", False)
        seed = attrs.get("seed", None)

        if not X:
            raise MlflowException(
                f"The provided prediction configuration pd.DataFrame columns ({df_schema}) \
                do not contain the required column `X` for specifying the regressor \
                values.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if not X_cols:
            raise MlflowException(
                f"The provided prediction configuration pd.DataFrame columns ({df_schema}) \
                do not contain the required column `X_cols` for specifying the \
                regressor columns.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Create Pandas DataFrame as required by Orbit predict method
        df = pd.DataFrame(data=X, columns=X_cols)

        # When the model is served via REST API the exogenous regressor must be
        # provided as a list to the configuration DataFrame to be JSON serializable
        # where the datetime entries of date_col are of string type. Below we convert
        # back the date_col to datetime format as required by orbit predict method.
        df[self.orbit_model.date_col] = df[self.orbit_model.date_col].astype(
            "datetime64[ns]"
        )

        predictions = self.orbit_model.predict(
            df,
            decompose=decompose,
            store_prediction_array=store_prediction_array,
            seed=seed,
        )

        return predictions
