Sktime
------

This example trains a ``Sktime`` NaiveForecaster model using the Longley dataset for
forecasting with exogenous variables.

Running the code
~~~~~~~~~~~~~~~~

Run the ``train.py`` module to create a new MLflow experiment (that logs the training
hyper-parameters, evaluation metrics and the trained model as an artifact) and to
compute interval forecasts loading the trained model in native flavor and ``pyfunc`` flavor:

.. include:: ../examples/sktime/train.py
   :code: python

To view the newly created experiment and logged artifacts open the MLflow UI:

.. code-block:: bash

    mlflow ui


Model serving
~~~~~~~~~~~~~

This section illustrates an example of serving the ``pyfunc`` flavor to a local REST
API endpoint and subsequently requesting a prediction from the served model. To serve
the model run the command below where you substitute the run id printed during execution
of the ``train.py`` module:

.. code-block:: bash

    mlflow models serve -m runs:/<run_id>/model --env-manager local --host 127.0.0.1

Open a new terminal and run the ``score_model.py`` module to request a prediction from the
served model (for more details read the
`Deploy MLflow models <https://mlflow.org/docs/latest/models.html#deploy-mlflow-models>`_ section in the official MLflow docs):

.. include:: ../examples/sktime/score_model.py
   :code: python
