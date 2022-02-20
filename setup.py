from setuptools import setup
from setuptools.command.build_py import build_py

setup(
    name="PyRisk",
    version="0.1.0",
    author="Laura Isaza Echeverri - Juan D. Velasquez",
    author_email="lisazae@unal.edu.co - jdvelasq@unal.edu.co",
    description="Risk Analysis, Through Simulation",
    long_description="Risk Analysis, Through Simulation",
    keywords="Risk Analysisy",
    provides=["Libreria"],
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "sympy",
        "matplotlib",
        "ipywidgets",
        "seaborn",
        "math"
    ],
)
