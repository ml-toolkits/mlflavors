SDV
---

This example trains a `SDV <https://github.com/sdv-dev/SDV>`_ SingleTablePreset
synthesizer model using a fake dataset. The fake dataset describes some fictional
guests staying at a hotel and the data is available as a single table.

Model logging and loading
~~~~~~~~~~~~~~~~~~~~~~~~~

Run the ``train.py`` module to create a new MLflow experiment (that logs the evaluation
metrics and the trained model as an artifact) and to generate synthetic data loading the
trained model in native flavor and ``pyfunc`` flavor:

.. include:: ../examples/sdv/train.py
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

Open a new terminal and run the ``score_model.py`` module to request a prediction from
the served model:

.. include:: ../examples/sdv/score_model.py
   :code: python
