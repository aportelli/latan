#!/usr/bin/env bash
set -euo pipefail

uv run python -m mkinit -i --recursive --nomods latan
find latan -name __init__.py -type f -exec uv run ruff check --fix {} +
find latan -name __init__.py -type f -exec uv run ruff format {} +
