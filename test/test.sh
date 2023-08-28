#!/bin/bash

cp ../source/pacmap/pacmap.py ./
rm -rf test_log
rm -rf test_output
mkdir test_log
mkdir test_output
python test_general.py > ./test_log/general.log 2> ./test_log/general.err
python test_randomness.py  > ./test_log/randomness.log 2> ./test_log/randomness.err
python test_metric.py  > ./test_log/metric.log 2> ./test_log/metric.err
python test_transform.py > ./test_log/transform.log 2> ./test_log/transform.err
python test_transform_tree.py > ./test_log/transform_tree.log 2> ./test_log/transform_tree.err
python test_transform_iris.py > ./test_log/transform_iris.log 2> ./test_log/transform_iris.err
