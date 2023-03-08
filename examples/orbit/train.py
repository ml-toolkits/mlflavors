import json

import mlflow
import pandas as pd
from orbit.models import ETS
from orbit.utils.dataset import load_iclaims
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import mlflow_flavors

ARTIFACT_PATH = "model"

with mlflow.start_run() as run:
    df = load_iclaims()
    date_col = "week"
    response_col = "claims"

    test_size = 52
    train_df = df[:-test_size]
    test_df = df[-test_size:]

    ets = ETS(
        response_col=response_col,
        date_col=date_col,
        seasonality=52,
        seed=8888,
    )
    ets.fit(df=train_df)

    # Extract parameters
    parameters = {
        k: ets.get_training_meta().get(k)
        for k in [
            "num_of_obs",
            "response_sd",
            "response_mean",
            "training_start",
            "training_end",
            "date_col",
            "response_col",
        ]
    }
    parameters["training_start"] = str(parameters["training_start"])
    parameters["training_end"] = str(parameters["training_end"])

    # Evaluate model
    y_pred = ets.predict(df=test_df, seed=2023)["prediction"]
    y_test = test_df["claims"]

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test, y_pred),
    }

    print(f"Parameters: \n{json.dumps(parameters, indent=2)}")
    print(f"Metrics: \n{json.dumps(metrics, indent=2)}")

    # Log parameters and metrics
    mlflow.log_params(parameters)
    mlflow.log_metrics(metrics)

    # Log model using pickle serialization (default).
    mlflow_flavors.orbit.log_model(
        orbit=ets,
        artifact_path=ARTIFACT_PATH,
        serialization_format="pickle",
    )
    model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)

# Load model in native orbit flavor and pyfunc flavor
loaded_model = mlflow_flavors.orbit.load_model(model_uri=model_uri)
loaded_pyfunc = mlflow_flavors.orbit.pyfunc.load_model(model_uri=model_uri)

# Convert test data to 2D numpy array so it can be passed to pyfunc predict using
# a single-row Pandas DataFrame configuration argument
X_test_array = test_df.to_numpy()

# Create configuration DataFrame for forecast with required parameters (X, X_cols),
# optional parameters, and regressor term.
predict_conf = pd.DataFrame(
    [
        {
            "X": X_test_array,
            "X_cols": test_df.columns,
            "decompose": True,
            "store_prediction_array": True,
            "seed": 2023,
        }
    ]
)

# Generate forecasts with native orbit flavor and pyfunc flavor
print(
    f"\nNative orbit 'predict':\n$ \
    {loaded_model.predict(test_df, decompose=True, store_prediction_array=True, seed=2023)}"  # noqa: 401
)
print(f"\nPyfunc 'predict':\n${loaded_pyfunc.predict(predict_conf)}")

# Print the run id wich is used for serving the model to a local REST API endpoint
# in the score_model.py module
print(f"\nMLflow run id:\n{run.info.run_id}")
