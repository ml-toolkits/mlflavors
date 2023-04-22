import pandas as pd
import requests

# Define local host and endpoint url
host = "127.0.0.1"
url = f"http://{host}:5000/invocations"

# Create configuration DataFrame
predict_conf = pd.DataFrame(
    [
        {
            "modality": "single_table",
            "num_rows": 10,
        }
    ]
)

# Create dictionary with pandas DataFrame in the split orientation
json_data = {"dataframe_split": predict_conf.to_dict(orient="split")}

# Score model
response = requests.post(url, json=json_data)
print(f"\nPyfunc sampling:\n${response.json()}")
