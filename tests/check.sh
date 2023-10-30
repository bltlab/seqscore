#!/usr/bin/env bash
set -euxo pipefail

files=(seqscore/ tests/ setup.py)
ruff check "${files[@]}"
mypy "${files[@]}"
