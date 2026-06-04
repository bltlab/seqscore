#!/usr/bin/env bash
set -euxo pipefail

flowmark --check ./*.md
files=(seqscore/ tests/)
ruff check "${files[@]}"
mypy "${files[@]}"
