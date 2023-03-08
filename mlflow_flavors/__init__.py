__version__ = "0.0.1"

try:
    from mlflow_flavors import orbit, sktime  # noqa: F401

    _model_flavors_supported = [
        "orbit",
        "sktime",
    ]
except ImportError as e:  # noqa: F841
    pass
