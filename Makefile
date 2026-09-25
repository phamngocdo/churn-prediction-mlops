.PHONY: install
install:
	uv venv --python 3.12
	uv pip install --no-cache -e ".[dev]"

.PHONY: lock
lock:
	uv pip freeze > requirements-lock.txt