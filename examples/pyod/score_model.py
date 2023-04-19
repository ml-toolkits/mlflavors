import pandas as pd
import requests
from pyod.utils.data import generate_data

contamination = 0.1  # percentage of outliers
n_train = 200  # number of training points
n_test = 100  # number of testing points

_, X_test, _, _ = generate_data(
    n_train=n_train, n_test=n_test, contamination=contamination
)

# Define local host and endpoint url
host = "127.0.0.1"
url = f"http://{host}:5000/invocations"

# Convert to list for JSON serialization
X_test_list = X_test.tolist()

# Create configuration DataFrame
predict_conf = pd.DataFrame(
    [
        {
            "X": X_test_list,
            "predict_method": "decision_function",
        }
    ]
)

# Create dictionary with pandas DataFrame in the split orientation
json_data = {"dataframe_split": predict_conf.to_dict(orient="split")}

# Score model
response = requests.post(url, json=json_data)
print(f"\nPyfunc 'decision_function':\n${response.json()}")
