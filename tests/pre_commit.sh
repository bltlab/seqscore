#!/usr/bin/env bash
set -euxo pipefail

files='seqscore/ tests/ *.py'
isort $files
black $files
flake8 $files
mypy $files
pytest --cov-report term-missing --cov=seqscore tests/
