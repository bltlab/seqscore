#! /usr/bin/env bash
set -euxo pipefail

rm -rf dist/*
python -m pip install --upgrade build twine
python -m build

# When the above is done, run the following.
# This is intentionally not run in the script.
# python -m twine upload dist/*
