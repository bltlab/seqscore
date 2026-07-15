#!/usr/bin/env bash
set -euxo pipefail

# See .coveragerc for configuration. Note that this will fail if
# coverage is below 100%.
pytest --cov --cov-report=term-missing tests/
