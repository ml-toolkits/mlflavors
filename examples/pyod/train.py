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

# Load model in native pyod flavor and pyfunc flavor
loaded_model = mlflavors.pyod.load_model(model_uri=model_uri)
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

# Generate anomaly scores with native pyod flavor and pyfunc flavor
print(
    f"\nNative pyod 'decision_function':\n$ \
    {loaded_model.decision_function(X_test)}"
)
print(f"\nPyfunc 'decision_function':\n${loaded_pyfunc.predict(predict_conf)[0]}")

# Print the run id wich is used for serving the model to a local REST API endpoint
# in the score_model.py module
print(f"\nMLflow run id:\n{run.info.run_id}")
