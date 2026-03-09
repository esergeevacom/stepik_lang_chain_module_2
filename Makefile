
init:
	pip install uv
	uv pip install -r build\requirements.txt


fmt:
	black --verbose --config pyproject.toml src
	isort --sp pyproject.toml src
	bandit -c pyproject.toml -r src


linters:
	black --check --config pyproject.toml src
	isort --sp pyproject.toml --check src
	bandit -c pyproject.toml -r src
	ruff --config pyproject.toml check src
