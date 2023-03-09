<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/mlflow-flavors.svg?branch=main)](https://cirrus-ci.com/github/<USER>/mlflow-flavors)
[![ReadTheDocs](https://readthedocs.org/projects/mlflow-flavors/badge/?version=latest)](https://mlflow-flavors.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/mlflow-flavors/main.svg)](https://coveralls.io/r/<USER>/mlflow-flavors)
[![PyPI-Server](https://img.shields.io/pypi/v/mlflow-flavors.svg)](https://pypi.org/project/mlflow-flavors/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/mlflow-flavors.svg)](https://anaconda.org/conda-forge/mlflow-flavors)
[![Monthly Downloads](https://pepy.tech/badge/mlflow-flavors/month)](https://pepy.tech/project/mlflow-flavors)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/mlflow-flavors)
-->

[![tests](https://github.com/blue-pen-labs/mlflow-flavors/actions/workflows/ci.yml/badge.svg)](https://github.com/blue-pen-labs/mlflow-flavors/actions/workflows/ci.yml)

# Mlflow-Flavors: A Python Package for hosting MLflow custom model flavors

This package adds MLflow support for some popular machine learning frameworks that are currently not available as MLflow built-in model flavors. It provides an intuitive design that closely follows the interface of MLflow built-in model flavors.

## Supported frameworks

| Framework | Links | Area |
|---|---|---|
| **Orbit** | [Example](examples/orbit/README.md) | Time Series
| **Sktime** | [Example](examples/sktime/README.md) | Time Series  ||

##  Installation

Installing from PyPI with all flavors:

```sh
$ pip install mlflow-flavors[all]
```
Installing from PyPI with a particular flavor:

```sh
$ pip install mlflow-flavors[orbit]
```

## Quick Start

Save an ``orbit`` ETS model as an artifact to MLflow:

```python
import mlflow_flavors

from orbit.models import ETS
from orbit.utils.dataset import load_iclaims

df = load_iclaims()

test_size = 52
train_df = df[:-test_size]
test_df = df[-test_size:]

ets = ETS(
    response_col="claims",
    date_col="week",
    seasonality=52,
    seed=8888,
)
ets.fit(df=train_df)

mlflow_flavors.orbit.save_model(
    orbit_model=ets,
    path="model",
)

loaded_model = mlflow_flavors.orbit.load_model("model")
loaded_model.predict(test_df)
```

Refer to the [examples](examples) folder for more extended usage examples for the individual flavors.

## Versioning

We document versions and changes in our [changelog](CHANGELOG.md).

## Local environment setup

Instantiate a local Python environment, for example:

```sh
$ conda create -n mlflow-flavors-dev python=3.9
$ source activate mlflow-flavors-dev
```
Install project locally:

```sh
$ python -m pip install --upgrade pip
$ pip install -e ".[dev,docs]"
```

Install pre-commit hooks:

```sh
$ pre-commit install
```

Build package documentation

```sh
$ cd docs
$ make html
```

Run tests

```sh
$ pytest tests/unit --cov
```
