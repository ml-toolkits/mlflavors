"""The file configures the Python package."""

from setuptools import find_packages, setup

from mlflavors.version import __version__

PACKAGE_REQUIREMENTS = [
    "datasetsforecast==0.0.8",
    "mlflow",
    "scikit-learn",
    "orbit-ml",
    "pmdarima",
    "sktime",
    "statsforecast",
    "pyod",
    "sdv",
]

DEV_REQUIREMENTS = [
    "pre-commit",
    "pytest",
    "pytest-cov",
    "setuptools",
    "tomli",
]

DOC_REQUIREMENTS = [
    "nbsphinx==0.9.1",
    "numpydoc==1.5.0",
    "sphinx_rtd_theme==1.1.1",
    "sphinx==5.3.0",
]

setup(
    name="mlflavors",
    description="""
        MLflavors: A collection of custom MLflow flavors.
        """,
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    author="Benjamin Bluhm",
    license="BSD-3-Clause",
    license_files="LICENSE.txt",
    url="https://github.com/ml-toolkits/mlflavors",
    project_urls={
        "Issue Tracker": "https://github.com/ml-toolkits/mlflavors/issues",
        "Documentation": "https://mlflavors.readthedocs.io/en/latest/",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    setup_requires=["setuptools", "wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "docs": DOC_REQUIREMENTS,
    },
    version=__version__,
    keywords="machine-learning ai mlflow",
    python_requires=">=3.7",
)
