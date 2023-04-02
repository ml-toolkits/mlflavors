# Orbit Example

This example trains an `orbit` Bayesian ETS model using the iclaims dataset which contains
the weekly initial claims for US unemployment benefits against a few related Google
trend queries from Jan 2010 - June 2018.

## Running the code

Run the `train.py` module to create a new MLflow experiment (that logs the training
hyper-parameters, evaluation metrics and the trained model as an artifact) and to
compute interval forecasts loading the trained model in native flavor and `pyfunc` flavor:

```
python train.py
```

To view the newly created experiment and logged artifacts open the MLflow UI:

```
mlflow ui
```

## Model serving

This section illustrates an example of serving the `pyfunc` flavor to a local REST
API endpoint and subsequently requesting a prediction from the served model. To serve
the model run the command below where you substitute the run id printed during execution
of the `train.py` module:

```
mlflow models serve -m runs:/<run_id>/model --env-manager local --host 127.0.0.1

```

Open a new terminal and run the `score_model.py` module to request a prediction from the
served model (for more details read the
[MLflow deployment API reference](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models)):

```
python score_model.py
```

## Running the code as a project

You can also run the code as a project as follows:

```
mlflow run .

```