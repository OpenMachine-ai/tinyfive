#!/bin/bash -e

# Make sure to increment the version number in pyproject.toml before
# running this script! See below link for more details:
# https://packaging.python.org/en/latest/tutorials/packaging-projects/

# copy all relevant files into folder pypi
rm -rf pypi
mkdir pypi
mkdir -p pypi/tinyfive
cp ../LICENSE ../README.md pyproject.toml pypi
cp ../machine.py                          pypi/tinyfive
touch                                     pypi/tinyfive/__init__.py

# build and upload
cd pypi
python3 -m build
python3 -m twine upload dist/*
