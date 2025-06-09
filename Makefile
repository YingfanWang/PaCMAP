.PHONY: install test clean

install:
	pip install -e .
	pip install pytest

test:
	pytest \
		test/test_general.py \
		test/test_transform_iris.py \
		test/test_randomness.py \
		test/test_transform.py \
		test/test_transform_tree.py \
		test/test_metric.py

clean:
	pip uninstall -y pacmap
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf source/*.egg-info/
