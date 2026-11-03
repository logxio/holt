#!/bin/sh
set -eu
for candidate in /opt/homebrew/bin/python3.13 python3 "${PYTHON:-python3}"; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import numpy" >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done
PYTHONPATH=src "${PY:-python3}" -m holt reproduce
