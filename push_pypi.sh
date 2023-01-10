#!/bin/bash -e

# Make sure to increment the version number in pyproject.toml before
# running this script! See below link for more details:
# https://packaging.python.org/en/latest/tutorials/packaging-projects/

# copy all relevant files into folder pypi
rm -rf pypi
mkdir pypi
mkdir -p pypi/src/tinyfive pypi/tests
cp LICENSE pyproject.toml README.md pypi
cp tinyfive.py                      pypi/src/tinyfive
touch                               pypi/src/tinyfive/__init__.py

# build and upload
cd pypi
python3 -m build
python3 -m twine upload dist/*
