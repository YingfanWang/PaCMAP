.PHONY: test

test:
	pytest test/test_general.py test/test_transform_iris.py test/test_randomness.py
