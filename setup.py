"""The file configures the Python package."""

from setuptools import find_packages, setup

from mlflow_flavors import __version__

# packages for local development and unit testing
PACKAGE_REQUIREMENTS = [
    "importlib-metadata",
    "mlflow-skinny",
    # "Cython>=0.22",
    # "numpy",
]

DEV_REQUIREMENTS = [
    "pre-commit",
    "pytest",
    "pytest-cov",
    "setuptools",
    "tomli",
]

DOC_REQUIREMENTS = [
    "sphinx_rtd_theme==1.1.1",
    "sphinx==5.3.0",
]

ORBIT_REQUIREMENTS = [
    "pystan @ git+https://github.com/stan-dev/pystan2.git@master#egg=pystan",
    "orbit-ml",
]

setup(
    name="mlflow_flavors",
    description="""
        This package provides MLflow custom flavors for
        some popular machine learning frameworks that
        are currently not available as MLflow built-in flavors.
        """,
    author="Benjamin Bluhm",
    license="BSD-3-Clause",
    license_files="LICENSE.txt",
    url="https://github.com/blue-pen-labs/mlflow-flavors",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["setuptools", "wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "docs": DOC_REQUIREMENTS,
        "orbit": ORBIT_REQUIREMENTS,
    },
    version=__version__,
    include_package_data=True,
)
