# File: setup.py
from setuptools import find_packages, setup

setup(
    name="excalibuhr",
    packages=find_packages(where="src"),
    version="0.1",
    package_dir={"": "src"},
)
