
MLflavors
=========

The MLflavors package adds MLflow support for some popular machine learning frameworks currently
not considered for inclusion as MLflow built-in flavors. You can use this package just like MLflow
built-in flavors to save and load your models. Some of the key benefits are listed below:

- Save your trained model as an MLflow artifact
- Load your model from MLflow for batch inference tasks
- Serve your model for real-time inference to an endpoint in the cloud
  (e.g. Databricks, Azure ML, AWS Sagemaker, etc.) using MLflow built-in deployment tools
- Get inspiration for creating your own MLflow custom flavor

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

The MLflow interface for the supported frameworks closely follows the design of built-in flavors.

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

Save an `Orbit <https://github.com/uber/orbit>`_ ETS model as an artifact to MLflow:

.. code-block:: python

    import mlflavors

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

    mlflavors.orbit.save_model(
        orbit_model=ets,
        path="model",
    )

Make a prediction loading the model from MLflow in native format:

.. code-block:: python

    loaded_model = mlflavors.orbit.load_model("model")
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

    loaded_pyfunc = mlflavors.orbit.pyfunc.load_model("model")
    loaded_pyfunc.predict(predict_conf)

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
    $ pip install -e ".[dev,docs]"

Install pre-commit hooks:

.. code-block:: bash

    $ pre-commit install

Run tests:

.. code-block:: bash

    $ pytest tests/unit --cov

Build Sphinx docs:

.. code-block:: bash

    $ cd docs
    $ make html
