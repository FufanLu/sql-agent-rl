PY ?= .venv/bin/python

.PHONY: dev test lint typecheck check push serve-sql

dev:            ## local CPU env for tests and the SQL service
	python3 -m venv .venv && .venv/bin/pip install -q uv && .venv/bin/uv pip install --python .venv/bin/python -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

typecheck:
	$(PY) -m mypy

check: lint typecheck test

push:           ## copy the code to the cluster (needs an open ssh master: ssh explorer true)
	bash cluster/push.sh

serve-sql:      ## the SQL sandbox service; SAR_DB_ROOT must point at the OmniSQL data/ dir
	$(PY) -m uvicorn sar.sql_env.server:app --app-dir src --port 8090
