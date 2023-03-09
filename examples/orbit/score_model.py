import pandas as pd
import requests
from orbit.utils.dataset import load_iclaims

df = load_iclaims()
test_size = 52
test_df = df[-test_size:]

# Define local host and endpoint url
host = "127.0.0.1"
url = f"http://{host}:5000/invocations"

# Convert DateTime to string for JSON serialization
test_df_pyfunc = test_df.copy()
test_df_pyfunc["week"] = test_df_pyfunc["week"].dt.strftime(
    date_format="%Y-%m-%d %H:%M:%S"
)

# Convert to list for JSON serialization
X_test_list = test_df_pyfunc.to_numpy().tolist()

# Convert index to list of strings for JSON serialization
X_cols = list(test_df.columns)

# Convert to dtypes to string for JSON serialization
X_dtypes = [str(dtype) for dtype in list(test_df.dtypes)]

predict_conf = pd.DataFrame(
    [
        {
            "X": X_test_list,
            "X_cols": X_cols,
            "X_dtypes": X_dtypes,
            "decompose": True,
            "store_prediction_array": True,
            "seed": 2023,
        }
    ]
)

# Create dictionary with pandas DataFrame in the split orientation
json_data = {"dataframe_split": predict_conf.to_dict(orient="split")}

# Score model
response = requests.post(url, json=json_data)
print(f"\nPyfunc 'predict_interval':\n${response.json()}")
