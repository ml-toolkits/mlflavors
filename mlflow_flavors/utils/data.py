import numpy as np
import pandas as pd


def load_m5(directory):
    """Adpat function from datasetsforecast.m5 module to filter down items."""
    path = f"{directory}/m5/datasets"

    # Calendar data
    cal_dtypes = {
        "wm_yr_wk": np.uint16,
        "event_name_1": "category",
        "event_type_1": "category",
        "event_name_2": "category",
        "event_type_2": "category",
        "snap_CA": np.uint8,
        "snap_TX": np.uint8,
        "snap_WI": np.uint8,
    }
    cal = pd.read_csv(
        f"{path}/calendar.csv",
        dtype=cal_dtypes,
        usecols=list(cal_dtypes.keys()) + ["date"],
        parse_dates=["date"],
    )
    cal["d"] = np.arange(cal.shape[0]) + 1
    cal["d"] = "d_" + cal["d"].astype("str")
    cal["d"] = cal["d"].astype("category")

    event_cols = [k for k in cal_dtypes if k.startswith("event")]
    for col in event_cols:
        cal[col] = cal[col].cat.add_categories("nan").fillna("nan")

    # Prices
    prices_dtypes = {
        "store_id": "category",
        "item_id": "category",
        "wm_yr_wk": np.uint16,
        "sell_price": np.float32,
    }

    prices = pd.read_csv(f"{path}/sell_prices.csv", dtype=prices_dtypes)

    # Sales
    sales_dtypes = {
        "item_id": prices.item_id.dtype,
        "dept_id": "category",
        "cat_id": "category",
        "store_id": "category",
        "state_id": "category",
        **{f"d_{i+1}": np.float32 for i in range(1969)},
    }
    # Reading train and test sets
    sales_train = pd.read_csv(f"{path}/sales_train_evaluation.csv", dtype=sales_dtypes)
    sales_test = pd.read_csv(f"{path}/sales_test_evaluation.csv", dtype=sales_dtypes)
    sales = sales_train.merge(
        sales_test,
        how="left",
        on=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
    )
    sales["id"] = (
        sales[["item_id", "store_id"]]
        .astype(str)
        .agg("_".join, axis=1)
        .astype("category")
    )
    sales = sales[sales["id"] == "FOODS_3_586_CA_3"]
    # Long format
    long = sales.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        var_name="d",
        value_name="y",
    )
    long["d"] = long["d"].astype(cal.d.dtype)
    long = long.merge(cal, on=["d"])
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"])
    long = long.drop(columns=["d", "wm_yr_wk"])

    def first_nz_mask(values, index):
        """Return a boolean mask where the True starts at the first non-zero value."""
        mask = np.full(values.size, True)
        for idx, value in enumerate(values):
            if value == 0:
                mask[idx] = False
            else:
                break
        return mask

    long = long.sort_values(["id", "date"], ignore_index=True)
    keep_mask = long.groupby("id")["y"].transform(first_nz_mask, engine="numba")
    long = long[keep_mask.astype(bool)]
    long.rename(columns={"id": "unique_id", "date": "ds"}, inplace=True)
    Y_df = long.filter(items=["unique_id", "ds", "y"])
    cats = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    S_df = long.filter(items=["unique_id"] + cats)
    S_df = S_df.drop_duplicates(ignore_index=True)
    X_df = long.drop(columns=["y"] + cats)

    Y_ts = Y_df.reset_index(drop=True)
    X_ts = X_df.reset_index(drop=True)
    X_ts = X_ts[["unique_id", "ds", "sell_price", "snap_CA"]]
    X_ts["unique_id"] = X_ts.unique_id.astype(str)

    # Extract dates for train and test set
    dates = Y_df["ds"].unique()
    dtrain = dates[:-28]  # noqa: F841
    dtest = dates[-28:]  # noqa: F841

    Y_train = Y_ts.query("ds in @dtrain")
    Y_test = Y_ts.query("ds in @dtest")

    X_train = X_ts.query("ds in @dtrain")  # noqa: F841
    X_test = X_ts.query("ds in @dtest")

    train_df = Y_train.merge(X_ts, how="left", on=["unique_id", "ds"])

    return train_df, X_test, Y_test
