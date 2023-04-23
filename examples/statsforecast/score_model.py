import pandas as pd
import requests

from mlflavors.utils.data import load_m5

DATA_PATH = "./data"
HORIZON = 28
LEVEL = [90, 95]
_, X_test, _ = load_m5(DATA_PATH)

# Define local host and endpoint url
host = "127.0.0.1"
url = f"http://{host}:5000/invocations"

# Convert DateTime to string for JSON serialization
X_test_pyfunc = X_test.copy()
X_test_pyfunc["ds"] = X_test_pyfunc["ds"].dt.strftime(date_format="%Y-%m-%d")

# Convert to list for JSON serialization
X_test_list = X_test_pyfunc.to_numpy().tolist()

# Convert index to list of strings for JSON serialization
X_cols = list(X_test.columns)

# Convert dtypes to string for JSON serialization
X_dtypes = [str(dtype) for dtype in list(X_test.dtypes)]

predict_conf = pd.DataFrame(
    [
        {
            "X": X_test_list,
            "X_cols": X_cols,
            "X_dtypes": X_dtypes,
            "h": HORIZON,
            "level": LEVEL,
        }
    ]
)

# Create dictionary with pandas DataFrame in the split orientation
json_data = {"dataframe_split": predict_conf.to_dict(orient="split")}

# Score model
response = requests.post(url, json=json_data)
print(f"\nPyfunc 'predict':\n${response.json()}")
