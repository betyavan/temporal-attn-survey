.PHONY: docker docker-push check refactor lint show install black mdformat yamlfix mypy ruff-lint ruff-fix mdlint yamllint smoke-test test clean dvc-init hf-cache-init

tag ?= latest
image ?= $(notdir $(shell pwd))

docker:
	docker build -t $(image):$(tag) -f Dockerfile .

docker-push:
	docker push $(image):$(tag)

check: refactor lint test clean

refactor: black mdformat yamlfix ruff-fix

lint: mypy mdlint ruff-lint yamllint

show:
	uv run python --version && uv pip list

install:
	uv sync

dvc-init:
	uv run dvc init

hf-cache-init:
	mkdir -p /tmp/hf_cache/hub /tmp/hf_cache/datasets
	@echo "HF cache prepared at /tmp/hf_cache"
	@echo "Activate it in your shell:  source ./set_cache_env.sh"

find_all_py = `find . -type f -name '*.py' | grep -v .venv | sort | uniq`
find_all_md = `find . -type f -name '*.md' | grep -v .venv | sort | uniq`
find_all_yaml = `find . -type f \( -iname \*.yaml -o -iname \*.yml \) | grep -v .venv | sort | uniq`

black:
	uv run black $(find_all_py)

mdformat:
	uv run mdformat $(find_all_md)

yamlfix:
	uv run yamlfix $(find_all_yaml)

yamllint:
	uv run yamlfix --check $(find_all_yaml)

mypy:
	uv run mypy --strict $(find_all_py) && rm -rf .mypy_cache

ruff-lint:
	uv run ruff check $(find_all_py)

ruff-fix:
	uv run ruff check --fix $(find_all_py)

mdlint:
	uv run mdformat --check $(find_all_md)

smoke-test:
	uv run pytest -vv -m fast tests && rm -rf .pytest_cache

test:
	uv run pytest -vv tests && rm -rf .pytest_cache

clean:
	uv run pyclean . && rm -rf __pycache__ && rm -rf *.egg-info && rm -rf build && rm -rf dist && rm -rf .pytest_cache && rm -rf .mypy_cache
