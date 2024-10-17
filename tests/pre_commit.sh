#!/usr/bin/env bash
set -euxo pipefail

files=(seqscore/ tests/ *.py)
ruff check --select I --fix "${files[@]}"  # Organize imports
ruff format "${files[@]}"
ruff check "${files[@]}"
mypy "${files[@]}"
pytest --cov-report term-missing --cov=seqscore tests/
