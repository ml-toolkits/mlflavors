"""
The ``mlflavors.sktime`` module provides an API for logging and loading sktime
models. This module exports sktime models with the following flavors:

sktime (native) format
    This is the main flavor that can be loaded back into sktime, which relies on pickle
    internally to serialize a model.

    Note that pickle serialization requires using the same python environment (version)
    in whatever environment you're going to use this model for inference to ensure that
    the model will load with appropriate version of pickle.

:py:mod:`mlflow.pyfunc` format
    Produced for use by generic pyfunc-based deployment tools and batch inference.

    The interface for utilizing a sktime model loaded as a ``pyfunc`` type for
    generating forecast predictions uses a *single-row* ``Pandas DataFrame``
    configuration argument. The following columns in this configuration
    ``Pandas DataFrame`` are supported:

    .. list-table::
      :widths: 15 10 15
      :header-rows: 1

      * - Column
        - Type
        - Description
      * - predict_method
        - str (required)
        - | Specifies the sktime predict method. The supported predict methods are
          | ``predict``, ``predict_interval``, ``predict_quantiles``, and
          | ``predict_var``.
      * - fh
        - list (optional)
        - | Specifies the number of future periods to generate starting from the last
          | datetime value of the training dataset, utilizing the frequency of the input
          | training series when the model was trained. (for example, if the training
          | data series elements represent one value per hour, in order to forecast 3
          | hours of future data, set the column ``fh`` to ``[1,2,3]``. If the parameter
          | is not provided it must be passed during :func:`fit()`.
          | (Default: ``None``)
      * - X_dtypes
        - numpy ndarray or list (optional)
        - | Exogenous regressor for future time period events.
          | For more information, read the underlying library explanation:
          | https://www.sktime.net/en/latest/examples/AA_datatypes_and_datasets.html#Section-1:-in-memory-data-containers.
      * - coverage
        - float (optional)
        - | The nominal coverage value for calculating prediction interval forecasts.
          | Can only be provided in combination with predict method
          | ``predict_interval``.
          | (Default: ``0.9``)
      * - alpha
        - float (optional)
        - | The probability value for calculating prediction quantile forecasts.
          | Can only be provided in combination with predict method
          | ``predict_quantiles``.
          | (Default: ``None``)
      * - cov
        - bool (optional)
        - | If True, computes covariance matrix forecast.
          | Can only be provided in combination with predict method ``predict_var``.
          | (Default: ``False``)

An example configuration for the ``pyfunc`` predict of a sktime model is shown below,
using an interval forecast with nominal coverage value ``[0.9,0.95]``, a future forecast
horizon of 3 periods, and no exogenous regressor elements:

====== ================= ============ ========
Index  predict_method    coverage     fh
====== ================= ============ ========
0      predict_interval  [0.9,0.95]   [1,2,3]
====== ================= ============ ========
"""  # noqa: E501
import logging
import os
import pickle

import mlflow
import numpy as np
import pandas as pd
import sktime
import yaml
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
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
from sktime.utils.multiindex import flatten_multiindex

import mlflavors

FLAVOR_NAME = "sktime"

SKTIME_PREDICT = "predict"
SKTIME_PREDICT_INTERVAL = "predict_interval"
SKTIME_PREDICT_QUANTILES = "predict_quantiles"
SKTIME_PREDICT_VAR = "predict_var"
SUPPORTED_SKTIME_PREDICT_METHODS = [
    SKTIME_PREDICT,
    SKTIME_PREDICT_INTERVAL,
    SKTIME_PREDICT_QUANTILES,
    SKTIME_PREDICT_VAR,
]

SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"
SUPPORTED_SERIALIZATION_FORMATS = [
    SERIALIZATION_FORMAT_PICKLE,
    SERIALIZATION_FORMAT_CLOUDPICKLE,
]

_logger = logging.getLogger(__name__)


def get_default_pip_requirements(include_cloudpickle=False):
    """
    :return: A list of default pip requirements for MLflow Models produced by this
             flavor. Calls to :func:`save_model()` and :func:`log_model()` produce a pip
             environment that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("sktime")]
    if include_cloudpickle:
        pip_deps += [_get_pinned_requirement("cloudpickle")]

    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(
        additional_pip_deps=get_default_pip_requirements(include_cloudpickle)
    )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    sktime_model,
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
    """
    Save a sktime model to a path on the local file system. Produces an MLflow Model
    containing the following flavors:

        - :py:mod:`mlflavors.sktime`
        - :py:mod:`mlflow.pyfunc`

    :param sktime_model: Fitted sktime model object.
    :param path: Local path where the model is to be saved.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or
        directories containing file dependencies). These files are *prepended* to the
        system path when the model is loaded.
    :param mlflow_model: mlflow.models.Model configuration to which to add the
        python_function flavor.
    :param signature: Model Signature mlflow.models.ModelSignature describes
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

        .. Warning:: If performing probabilistic forecasts (``predict_interval``,
          ``predict_quantiles``) with a ``sktime`` model, the signature
          on the returned prediction object will not be correctly inferred due
          to the Pandas MultiIndex column type when using the these methods.
          ``infer_schema`` will function correctly if using the ``pyfunc`` flavor
          of the model, though.
    :param input_example: Input example provides one or several instances of valid model
        input. The example can be used as a hint of what data to feed the model. The
        given example will be converted to a ``Pandas DataFrame`` and then serialized to
        json using the ``Pandas`` split-oriented format. Bytes are base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param serialization_format: The format in which to serialize the model. This should
        be one of the formats "pickle" or "cloudpickle"
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                f"Unrecognized serialization format: {serialization_format}. "
                "Please specify one of the following supported formats: "
                "{SUPPORTED_SERIALIZATION_FORMATS}."
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
    _save_model(
        sktime_model, model_data_path, serialization_format=serialization_format
    )

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflavors.sktime",
        model_path=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        sktime_version=sktime.__version__,
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
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs
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
    sktime_model,
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
    Log a sktime model as an MLflow artifact for the current run. Produces an MLflow
    Model containing the following flavors:

        - :py:mod:`mlflavors.sktime`
        - :py:mod:`mlflow.pyfunc`

    :param sktime_model: Fitted sktime model object.
    :param artifact_path: Run-relative artifact path to save the model instance to.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or
        directories containing file dependencies). These files are *prepended* to the
        system path when the model is loaded.
    :param registered_model_name: This argument may change or be removed in a future
        release without warning. If given, create a model version under
        ``registered_model_name``, also creating a registered model if one with the
        given name does not exist.
    :param signature: Model Signature mlflow.models.ModelSignature describes
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

        .. Warning:: If performing probabilistic forecasts (``predict_interval``,
          ``predict_quantiles``) with a ``sktime`` model, the signature
          on the returned prediction object will not be correctly inferred due
          to the Pandas MultiIndex column type when using the these methods.
          ``infer_schema`` will function correctly if using the ``pyfunc`` flavor
          of the model, though.
    :param input_example: Input example provides one or several instances of valid model
        input. The example can be used as a hint of what data to feed the model. The
        given example will be converted to a ``Pandas DataFrame`` and then serialized to
        json using the ``Pandas`` split-oriented format. Bytes are base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to
        finish being created and is in ``READY`` status. By default, the function waits
        for five minutes. Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param serialization_format: The format in which to serialize the model. This should
        be one of the formats "pickle" or "cloudpickle"

    :return: A :py:class:`ModelInfo` instance that contains the metadata of the logged
        model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflavors.sktime,
        registered_model_name=registered_model_name,
        sktime_model=sktime_model,
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
    Load a sktime model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts
                      <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: A sktime model.
    """  # noqa: E501
    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path
    )
    flavor_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME
    )
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    sktime_model_file_path = os.path.join(
        local_model_path, flavor_conf["pickled_model"]
    )
    serialization_format = flavor_conf.get(
        "serialization_format", SERIALIZATION_FORMAT_PICKLE
    )
    return _load_model(
        path=sktime_model_file_path, serialization_format=serialization_format
    )


def _save_model(model, path, serialization_format):
    with open(path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            pickle.dump(model, out)
        else:
            import cloudpickle

            cloudpickle.dump(model, out)


def _load_model(path, serialization_format):
    with open(path, "rb") as pickled_model:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            return pickle.load(pickled_model)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            return cloudpickle.load(pickled_model)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    :param path: Local filesystem path to the MLflow Model with the sktime flavor.
    """
    try:
        sktime_flavor_conf = _get_flavor_configuration(
            model_path=path, flavor_name=FLAVOR_NAME
        )
        serialization_format = sktime_flavor_conf.get(
            "serialization_format", SERIALIZATION_FORMAT_PICKLE
        )
    except MlflowException:
        _logger.warning(
            "Could not find sktime flavor configuration during model "
            "loading process. Assuming 'pickle' serialization format."
        )
        serialization_format = SERIALIZATION_FORMAT_PICKLE

    pyfunc_flavor_conf = _get_flavor_configuration(
        model_path=path, flavor_name=pyfunc.FLAVOR_NAME
    )
    path = os.path.join(path, pyfunc_flavor_conf["model_path"])

    return _SktimeModelWrapper(
        _load_model(path, serialization_format=serialization_format)
    )


class _SktimeModelWrapper:
    def __init__(self, sktime_model):
        self.sktime_model = sktime_model

    def predict(self, dataframe) -> pd.DataFrame:
        df_schema = dataframe.columns.values.tolist()

        if len(dataframe) > 1:
            raise MlflowException(
                f"The provided prediction pd.DataFrame contains {len(dataframe)} rows. "
                "Only 1 row should be supplied.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Convert the configuration dataframe into a dictionary to simplify the
        # extraction of parameters passed to the sktime predcition methods.
        attrs = dataframe.to_dict(orient="index").get(0)
        predict_method = attrs.get("predict_method")

        if not predict_method:
            raise MlflowException(
                f"The provided prediction configuration pd.DataFrame columns ({df_schema}) \
                do not contain the required column `predict_method` for specifying the \
                prediction method.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if predict_method not in SUPPORTED_SKTIME_PREDICT_METHODS:
            raise MlflowException(
                "Invalid `predict_method` value."
                f"The supported prediction methods are \
                {SUPPORTED_SKTIME_PREDICT_METHODS}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # For inference parameters 'fh', 'X', 'coverage', 'alpha', and 'cov'
        # the respective sktime default value is used if the value was not
        # provided in the configuration dataframe.
        fh = attrs.get("fh", None)

        # Any model that is trained with exogenous regressor elements will need
        # to provide `X` entries as a numpy ndarray to the predict method.
        X = attrs.get("X", None)

        # When the model is served via REST API the exogenous regressor must be
        # provided as a list to the configuration DataFrame to be JSON serializable.
        # Below we convert the list back to ndarray type as required by sktime
        # predict methods.
        if isinstance(X, list):
            X = np.array(X)

        if predict_method == SKTIME_PREDICT:
            predictions = self.sktime_model.predict(fh=fh, X=X)

        if predict_method == SKTIME_PREDICT_INTERVAL:
            coverage = attrs.get("coverage", 0.9)
            predictions = self.sktime_model.predict_interval(
                fh=fh, X=X, coverage=coverage
            )

        if predict_method == SKTIME_PREDICT_QUANTILES:
            alpha = attrs.get("alpha", None)
            predictions = self.sktime_model.predict_quantiles(fh=fh, X=X, alpha=alpha)

        if predict_method == SKTIME_PREDICT_VAR:
            cov = attrs.get("cov", False)
            predictions = self.sktime_model.predict_var(fh=fh, X=X, cov=cov)

        # Methods predict_interval() and predict_quantiles() return a pandas
        # MultiIndex column structure. As MLflow signature inference does not
        # support MultiIndex column structure the columns must be flattened.
        if predict_method in [SKTIME_PREDICT_INTERVAL, SKTIME_PREDICT_QUANTILES]:
            predictions.columns = flatten_multiindex(predictions)

        return predictions
