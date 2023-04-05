"""The file configures the Python package."""

from setuptools import find_packages, setup

from mlflow_flavors.version import __version__

# packages for local development and unit testing
PACKAGE_REQUIREMENTS = [
    "importlib-metadata",
    "numpy",
    "mlflow",
    "scikit-learn",
]

DEV_REQUIREMENTS = [
    "pre-commit",
    "pytest",
    "pytest-cov",
    "setuptools",
    "tomli",
    "pmdarima",
]

DOC_REQUIREMENTS = [
    "nbsphinx==0.9.1",
    "numpydoc==1.5.0",
    "sphinx_rtd_theme==1.1.1",
    "sphinx==5.3.0",
]

ORBIT_REQUIREMENTS = [
    "orbit-ml",
]

SKTIME_REQUIREMENTS = [
    "sktime",
]

setup(
    name="mlflow_flavors",
    description="""
        mlflow-flavors: A repository for hosting MLflow custom flavors.
        """,
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    author="Blue Pen Labs",
    license="BSD-3-Clause",
    license_files="LICENSE.txt",
    url="https://github.com/blue-pen-labs/mlflow-flavors",
    project_urls={
        "Issue Tracker": "https://github.com/blue-pen-labs/mlflow-flavors/issues",
        "Documentation": "https://mlflow-flavors.readthedocs.io/en/latest/",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    setup_requires=["setuptools", "wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "docs": DOC_REQUIREMENTS,
        "orbit": ORBIT_REQUIREMENTS,
        "sktime": SKTIME_REQUIREMENTS,
    },
    version=__version__,
    keywords="machine-learning ai mlflow",
    python_requires=">=3.7",
)
