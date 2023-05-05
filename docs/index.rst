Welcome to MLflavors` Documentation
===================================
The `MLflavors <https://github.com/ml-toolkits/mlflavors>`_ package adds MLflow support for some popular machine learning frameworks currently
not considered for inclusion as MLflow built-in flavors. You can use this package just like MLflow
built-in flavors to save and load your models. Some of the key benefits are listed below:

- Save your trained model as an MLflow artifact
- Load your model from MLflow for batch inference tasks
- Serve your model for real-time inference to an endpoint in the cloud
  (e.g. Databricks, Azure ML, AWS Sagemaker, etc.) using MLflow built-in deployment tools
- Get inspiration for creating your own MLflow custom flavor

The MLflow interface for the supported frameworks closely follows the design of built-in flavors.
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


.. toctree::
   :maxdepth: 2

   readme
   examples
   api
   changelog
