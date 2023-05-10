Welcome to MLflavors` Documentation
===================================
The `MLflavors <https://github.com/ml-toolkits/mlflavors>`_ package adds MLflow support for some popular machine learning frameworks currently
not considered for inclusion as MLflow built-in flavors. Just like built-in flavors, you can use
this package to save your model as an MLflow artifact, load your model from MLflow for batch
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

The MLflow interface for the supported frameworks closely follows the design of built-in flavors.
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

.. toctree::
   :maxdepth: 2

   readme
   examples
   api
   changelog
