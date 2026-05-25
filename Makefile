.PHONY: export infer train docker docker-push check refactor lint show install black mdformat yamlfix mypy pylint mdlint test clean docs docs-serve

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
	poetry run python --version && poetry show

prepare-environment:
	pip install -U pip
	pip install virtualenv
	virtualenv .venv
	. .venv/bin/activate && pip install -U pip && pip install poetry && poetry config virtualenvs.create false

install:
	poetry install --no-root

dvc-init:
	dvc init

find_all_py = `find . -type f -name '*.py' | grep -v .venv | sort | uniq`
find_all_md = `find . -type f -name '*.md' | grep -v .venv | sort | uniq`
find_all_yaml = `find . -type f \( -iname \*.yaml -o -iname \*.yml \) | grep -v .venv | sort | uniq`

black:
	poetry run black $(find_all_py)

mdformat:
	poetry run mdformat $(find_all_md)

yamlfix:
	poetry run yamlfix $(find_all_yaml)

yamllint:
	poetry run yamlfix --check $(find_all_yaml)

mypy:
	poetry run mypy --strict $(find_all_py) && rm -rf .mypy_cache

ruff-lint:
	poetry run ruff check $(find_all_py)

ruff-fix:
	poetry run ruff check --fix $(find_all_py)

mdlint:
	poetry run mdformat --check $(find_all_md)

smoke-test:
	poetry run pytest -vv -m fast tests && rm -rf .pytest_cache

test:
	poetry run pytest -vv  tests && rm -rf .pytest_cache

clean:
	poetry run pyclean . && rm -rf __pycache__ && rm -rf *.egg-info && rm -rf build && rm -rf dist && rm -rf .pytest_cache && rm -rf .mypy_cache
