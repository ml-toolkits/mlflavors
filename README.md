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

# mlflow-flavors

> This package provides MLflow custom flavors for some popular machine learning frameworks that are currently not available as MLflow built-in flavors.

## Local environment setup

1. Instantiate a local Python environment via a tool of your choice. This example is based on `conda`, but you can use any environment management tool:
```bash
conda create -n mlflow-flavors-dev python=3.9
source activate mlflow-flavors-dev
```

2. Install project locally:
```bash
pip install -e ".[dev,docs]"
```

3. Install pre-commit hooks
```bash
pre-commit install
```

## Building package documentation

Go to docs folder and run:
```
make html
```
View docs by opening `docs/build/html/index.html` in your browser.

Clean up Sphinx build resources:
```
clean html
```

## Running tests

Use `pytest`:
```
pytest tests/unit --cov
```
