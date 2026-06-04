#!/usr/bin/env bash
set -euxo pipefail

files=(seqscore/ tests/)
ruff check "${files[@]}"
mypy "${files[@]}"
