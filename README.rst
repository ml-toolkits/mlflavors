
mlflow-flavors
==============

This package adds MLflow support for some popular machine learning frameworks currently
not available as MLflow built-in flavors. The MLflow interface for the supported
frameworks follows the design of built-in flavors.

Using this package you can save your model as an MLflow artifact, load your model from
MLflow for batch inference, or serve your model for real-time inference to a REST API
endpoint.


Supported frameworks
--------------------
The following open-source libraris are currently supported:

    .. list-table::
      :widths: 15 10 15
      :header-rows: 1

      * - Framework
        - Quickstart
        - Category
      * - `Orbit <https://github.com/uber/orbit>`_
        - `MLflow-Orbit Example <https://mlflow-flavors.readthedocs.io/en/latest/examples.html#orbit>`_
        - Time Series Forecasting
      * - `Sktime <https://github.com/sktime/sktime>`_
        - `MLflow-Sktime Example <https://mlflow-flavors.readthedocs.io/en/latest/examples.html#sktime>`_
        - Time Series Forecasting
      * - `StatsForecast <https://github.com/Nixtla/statsforecast>`_
        - `MLflow-StatsForecast Example <https://mlflow-flavors.readthedocs.io/en/latest/examples.html#statsforecast>`_
        - Time Series Forecasting
      * - `PyOD <https://github.com/yzhao062/pyod>`_
        - `MLflow-PyOD Example <https://mlflow-flavors.readthedocs.io/en/latest/examples.html#pyod>`_
        - Anomaly Detection

Installation
------------

Installing from PyPI with all flavors:

.. code-block:: bash

   $ pip install mlflow-flavors[all]

Installing a particular flavor:

.. code-block:: bash

    $ pip install mlflow-flavors[orbit]

Quickstart
----------

Save an ``orbit`` ETS model as an artifact to MLflow:

.. code-block:: python

    import mlflow_flavors

    from orbit.models import ETS
    from orbit.utils.dataset import load_iclaims

    df = load_iclaims()

    test_size = 52
    train_df = df[:-test_size]
    test_df = df[-test_size:]

    ets = ETS(
        response_col="claims",
        date_col="week",
        seasonality=52,
        seed=8888,
    )
    ets.fit(df=train_df)

    mlflow_flavors.orbit.save_model(
        orbit_model=ets,
        path="model",
    )

Make a prediction loading the model from MLflow in native format:

.. code-block:: python

    loaded_model = mlflow_flavors.orbit.load_model("model")
    loaded_model.predict(test_df, decompose=True, store_prediction_array=True, seed=2023)

Make a prediction loading the model from MLflow in ``pyfunc`` format:

.. code-block:: python

    # Convert test data to 2D numpy array so it can be passed to pyfunc predict using
    # a single-row Pandas DataFrame configuration argument
    X_test_array = test_df.to_numpy()

    # Create configuration DataFrame
    predict_conf = pd.DataFrame(
        [
            {
                "X": X_test_array,
                "X_cols": test_df.columns,
                "X_dtypes": list(test_df.dtypes),
                "decompose": True,
                "store_prediction_array": True,
                "seed": 2023,
            }
        ]
    )

    loaded_pyfunc = mlflow_flavors.orbit.pyfunc.load_model("model")
    loaded_pyfunc.predict(predict_conf)

Documentation
-------------

Documentation, examples, and API reference for mlflow-flavors can be found
`here <https://mlflow-flavors.readthedocs.io/en/latest/index.html>`_.

Contributing
------------

We welcome contributions from the community. To open an issue or request a new feature,
please open a GitHub issue.

Versioning
----------

We document versions and changes in our
`changelog <https://github.com/blue-pen-labs/mlflow-flavors/tree/main/CHANGELOG.rst>`_ .

Development
-----------

To set up your local development environment, create a virtual environment, such as:

.. code-block:: bash

    $ conda create -n mlflow-flavors-dev python=3.9
    $ source activate mlflow-flavors-dev

Install project locally:

.. code-block:: bash

    $ python -m pip install --upgrade pip
    $ pip install -e ".[dev,docs]"

Install pre-commit hooks:

.. code-block:: bash

    $ pre-commit install

Run tests:

.. code-block:: bash

    $ pytest tests/unit --cov

Build package documentation:

.. code-block:: bash

    $ cd docs
    $ make html
