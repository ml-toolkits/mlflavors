import mlflow
import pandas as pd
from datasetsforecast.m5 import M5
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

import mlflavors
from mlflavors.utils.data import load_m5

ARTIFACT_PATH = "model"
DATA_PATH = "./data"
HORIZON = 28
LEVEL = [90, 95]

with mlflow.start_run() as run:
    M5.download(DATA_PATH)
    train_df, X_test, Y_test = load_m5(DATA_PATH)

    models = [AutoARIMA(season_length=7)]

    sf = StatsForecast(df=train_df, models=models, freq="D", n_jobs=-1)

    sf.fit()

    # Evaluate model
    y_pred = sf.predict(h=HORIZON, X_df=X_test, level=LEVEL)["AutoARIMA"]
    y_test = Y_test["y"]

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test, y_pred),
    }

    print(f"Metrics: \n{metrics}")

    # Log metrics
    mlflow.log_metrics(metrics)

    # Log model using pickle serialization (default).
    mlflavors.statsforecast.log_model(
        statsforecast_model=sf,
        artifact_path=ARTIFACT_PATH,
        serialization_format="pickle",
    )
    model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)

# Load model in native statsforecast flavor and pyfunc flavor
loaded_model = mlflavors.statsforecast.load_model(model_uri=model_uri)
loaded_pyfunc = mlflavors.statsforecast.pyfunc.load_model(model_uri=model_uri)

# Convert test data to 2D numpy array so it can be passed to pyfunc predict using
# a single-row Pandas DataFrame configuration argument
X_test_array = X_test.to_numpy()

# Create configuration DataFrame
predict_conf = pd.DataFrame(
    [
        {
            "X": X_test_array,
            "X_cols": X_test.columns,
            "X_dtypes": list(X_test.dtypes),
            "h": HORIZON,
            "level": LEVEL,
        }
    ]
)

# Generate forecasts with native statsforecast flavor and pyfunc flavor
print(
    f"\nNative statsforecast 'predict':\n$ \
    {loaded_model.predict(h=HORIZON, X_df=X_test, level=LEVEL)}"  # noqa: 401
)
print(f"\nPyfunc 'predict':\n${loaded_pyfunc.predict(predict_conf)}")

# Print the run id wich is used for serving the model to a local REST API endpoint
# in the score_model.py module
print(f"\nMLflow run id:\n{run.info.run_id}")
