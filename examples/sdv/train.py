import mlflow
import pandas as pd
from sdv.datasets.demo import download_demo
from sdv.evaluation.single_table import evaluate_quality
from sdv.lite import SingleTablePreset

import mlflow_flavors

ARTIFACT_PATH = "model"

with mlflow.start_run() as run:
    real_data, metadata = download_demo(
        modality="single_table", dataset_name="fake_hotel_guests"
    )

    # Train synthesizer
    synthesizer = SingleTablePreset(metadata, name="FAST_ML")
    synthesizer.fit(real_data)

    # Evaluate model
    synthetic_data = synthesizer.sample(num_rows=10)
    quality_report = evaluate_quality(
        real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
    )

    metrics = {
        "overall_quality_score": quality_report.get_score(),
    }

    # Log metrics
    mlflow.log_metrics(metrics)

    # Log model using pickle serialization (default).
    mlflow_flavors.sdv.log_model(
        sdv_model=synthesizer,
        artifact_path=ARTIFACT_PATH,
        serialization_format="pickle",
    )
    model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)

# Load model in native sdv flavor and pyfunc flavor
loaded_model = mlflow_flavors.sdv.load_model(model_uri=model_uri)
loaded_pyfunc = mlflow_flavors.sdv.pyfunc.load_model(model_uri=model_uri)

# Create configuration DataFrame
predict_conf = pd.DataFrame(
    [
        {
            "modality": "single_table",
            "num_rows": 10,
        }
    ]
)

# Generate synthetic data with native sdv flavor and pyfunc flavor
print(
    f"\nNative sdv sampling:\n$ \
    {loaded_model.sample(num_rows=10)}"
)
print(f"\nPyfunc sampling:\n${loaded_pyfunc.predict(predict_conf)}")

# Print the run id wich is used for serving the model to a local REST API endpoint
# in the score_model.py module
print(f"\nMLflow run id:\n{run.info.run_id}")
