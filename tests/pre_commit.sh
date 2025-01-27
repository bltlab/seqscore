#!/usr/bin/env bash
set -euxo pipefail

files=(seqscore/ tests/ *.py)
ruff check --fix "${files[@]}"
ruff check --select I --fix "${files[@]}"  # Organize imports
ruff format "${files[@]}"
ruff check "${files[@]}"  # Redundant but ensures CI will pass
mypy "${files[@]}"
pytest --cov-report term-missing --cov=seqscore tests/
