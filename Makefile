.PHONY: check-env test lint

PY ?= $(if $(wildcard .venv-app/bin/python),.venv-app/bin/python,python3)

check-env:
	$(PY) scripts/check_env.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
