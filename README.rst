
MLflavors
=========

The MLflavors package adds MLflow support for some popular machine learning frameworks currently
not considered for inclusion as MLflow built-in flavors. Similar to the built-in flavors, you can
use this package to save your model as an MLflow artifact, load your model from MLflow for batch
inference, and deploy your model to a serving endpoint using MLflow deployment tools.

The following open-source libraries are currently supported:

    .. list-table::
      :widths: 15 10 15
      :header-rows: 1

      * - Framework
        - Tutorials
        - Category
      * - `Orbit <https://github.com/uber/orbit>`_
        - `MLflow-Orbit <https://mlflavors.readthedocs.io/en/latest/examples.html#orbit>`_
        - Time Series Forecasting
      * - `Sktime <https://github.com/sktime/sktime>`_
        - `MLflow-Sktime <https://mlflavors.readthedocs.io/en/latest/examples.html#sktime>`_
        - Time Series Forecasting
      * - `StatsForecast <https://github.com/Nixtla/statsforecast>`_
        - `MLflow-StatsForecast <https://mlflavors.readthedocs.io/en/latest/examples.html#statsforecast>`_
        - Time Series Forecasting
      * - `PyOD <https://github.com/yzhao062/pyod>`_
        - `MLflow-PyOD <https://mlflavors.readthedocs.io/en/latest/examples.html#pyod>`_
        - Anomaly Detection
      * - `SDV <https://github.com/sdv-dev/SDV>`_
        - `MLflow-SDV <https://mlflavors.readthedocs.io/en/latest/examples.html#sdv>`_
        - Synthetic Data Generation

The interface design for the supported frameworks is similar to many of the existing built-in flavors.
Particularly, the interface for utilizing the custom model loaded as a ``pyfunc`` flavor
for generating predictions uses a single-row Pandas DataFrame configuration argument to expose the
parameters of the flavor's inference API.

|tests| |coverage| |docs| |pypi| |license|

.. |tests| image:: https://img.shields.io/github/actions/workflow/status/ml-toolkits/mlflavors/ci.yml?style=for-the-badge&logo=github
    :target: https://github.com/ml-toolkits/mlflavors/actions/workflows/ci.yml/

.. |coverage| image:: https://img.shields.io/codecov/c/github/ml-toolkits/mlflavors?style=for-the-badge&label=codecov&logo=codecov
    :target: https://codecov.io/gh/ml-toolkits/mlflavors

.. |docs| image:: https://img.shields.io/readthedocs/mlflavors/latest.svg?style=for-the-badge&logoColor=white
    :target: https://mlflavors.readthedocs.io/en/latest/index.html
    :alt: Latest Docs

.. |pypi| image:: https://img.shields.io/pypi/v/mlflavors.svg?style=for-the-badge&logo=pypi&logoColor=white
    :target: https://pypi.org/project/mlflavors/
    :alt: Latest Python Release

.. |license| image:: https://img.shields.io/badge/License-BSD--3--Clause-blue?style=for-the-badge
    :target: https://opensource.org/license/bsd-3-clause/
    :alt: BSD-3-Clause License

Documentation
-------------

Usage examples for all flavors and the API reference can be found in the package
`documenation <https://mlflavors.readthedocs.io/en/latest/index.html>`_.


Installation
------------

Installing from PyPI:

.. code-block:: bash

   $ pip install mlflavors

Quickstart
----------

This example trains a `PyOD <https://github.com/yzhao062/pyod>`_ KNN outlier detection
model using a synthetic dataset. A new MLflow experiment is created to log the evaluation
metrics and the trained model as an artifact and anomaly scores are computed loading the
trained model in native flavor and ``pyfunc`` flavor. Finally, the model is served
for real-time inference to a local endpoint.

Saving the model as an MLflow artifact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import json

    import mlflow
    import pandas as pd
    from pyod.models.knn import KNN
    from pyod.utils.data import generate_data
    from sklearn.metrics import roc_auc_score

    import mlflavors

    ARTIFACT_PATH = "model"

    with mlflow.start_run() as run:
        contamination = 0.1  # percentage of outliers
        n_train = 200  # number of training points
        n_test = 100  # number of testing points

        X_train, X_test, _, y_test = generate_data(
            n_train=n_train, n_test=n_test, contamination=contamination
        )

        # Train kNN detector
        clf = KNN()
        clf.fit(X_train)

        # Evaluate model
        y_test_scores = clf.decision_function(X_test)

        metrics = {
            "roc": roc_auc_score(y_test, y_test_scores),
        }

        print(f"Metrics: \n{json.dumps(metrics, indent=2)}")

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model using pickle serialization (default).
        mlflavors.pyod.log_model(
            pyod_model=clf,
            artifact_path=ARTIFACT_PATH,
            serialization_format="pickle",
        )
        model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)

    # Print the run id wich is used below for serving the model to a local REST API endpoint
    print(f"\nMLflow run id:\n{run.info.run_id}")

Loading the model from MLflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make a prediction loading the model from MLflow in native format:

.. code-block:: python

    loaded_model = mlflavors.pyod.load_model(model_uri=model_uri)
    print(loaded_model.decision_function(X_test))

Make a prediction loading the model from MLflow in ``pyfunc`` format:

.. code-block:: python

    loaded_pyfunc = mlflavors.pyod.pyfunc.load_model(model_uri=model_uri)

    # Create configuration DataFrame
    predict_conf = pd.DataFrame(
        [
            {
                "X": X_test,
                "predict_method": "decision_function",
            }
        ]
    )

    print(loaded_pyfunc.predict(predict_conf)[0])

Serving the model to an endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To serve the model to a local REST API endpoint run the command below where you substitute
the run id printed above:

.. code-block:: bash

    mlflow models serve -m runs:/<run_id>/model --env-manager local --host 127.0.0.1

Similarly, you could serve the model to an endpoint in the cloud (e.g. Azure ML, AWS SageMaker, etc.)
using `MLflow deployment tools <https://mlflow.org/docs/latest/models.html#built-in-deployment-tools>`_.
Open a new terminal and run the below model scoring script to request a prediction from
the served model:

.. code-block:: python

    import pandas as pd
    import requests
    from pyod.utils.data import generate_data

    contamination = 0.1  # percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

    _, X_test, _, _ = generate_data(
        n_train=n_train, n_test=n_test, contamination=contamination
    )

    # Define local host and endpoint url
    host = "127.0.0.1"
    url = f"http://{host}:5000/invocations"

    # Convert to list for JSON serialization
    X_test_list = X_test.tolist()

    # Create configuration DataFrame
    predict_conf = pd.DataFrame(
        [
            {
                "X": X_test_list,
                "predict_method": "decision_function",
            }
        ]
    )

    # Create dictionary with pandas DataFrame in the split orientation
    json_data = {"dataframe_split": predict_conf.to_dict(orient="split")}

    # Score model
    response = requests.post(url, json=json_data)
    print(response.json())

Contributing
------------

Contributions from the community are welcome, I will be happy to support the inclusion
and development of new features and flavors. To open an issue or request a new feature, please
open a GitHub issue.

Versioning
----------

Versions and changes are documented in the
`changelog <https://github.com/ml-toolkits/mlflavors/tree/main/CHANGELOG.rst>`_ .

Development
-----------

To set up your local development environment, create a virtual environment, such as:

.. code-block:: bash

    $ conda create -n mlflavors-dev python=3.9
    $ source activate mlflavors-dev

Install project locally:

.. code-block:: bash

    $ python -m pip install --upgrade pip
    $ pip install -e ".[dev]"

Install pre-commit hooks:

.. code-block:: bash

    $ pre-commit install

Run tests:

.. code-block:: bash

    $ pytest

Build Sphinx docs:

.. code-block:: bash

    $ cd docs
    $ make html
