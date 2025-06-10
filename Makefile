.PHONY: install test clean

install-dev:
	pip install -e .
	pip install -r requirements-test.txt

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
	rm -rf test/output/
