.PHONY: install test clean

requirements-test.txt:
	uv pip compile --group dev pyproject.toml -o requirements-test.txt

requirements.txt:
	uv pip compile pyproject.toml -o requirements.txt

install-dev:
	uv sync --locked --all-extras --dev

install:
	uv sync --locked

test:
	uv run pytest \
		test/test_general.py \
		test/test_transform_iris.py \
		test/test_randomness.py \
		test/test_transform.py \
		test/test_transform_tree.py \
		test/test_metric.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf source/*.egg-info/
	rm -rf test/output/

build:
	uv build
