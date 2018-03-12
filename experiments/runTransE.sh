#!/usr/bin/env bash

rm -rf build-te
mkdir build-te
cd build-te
cmake ../../../../
make

#train_file="../../../../data/WN18/wordnet-mlj12-train.txt"
#test_file="../../../../data/WN18/wordnet-mlj12-test.txt"
#valid_file="../../../../data/WN18/wordnet-mlj12-valid.txt"

train_file="../../../../data/FB15k/freebase_mtr100_mte100-train.txt"
test_file="../../../../data/FB15k/freebase_mtr100_mte100-test.txt"
valid_file="../../../../data/FB15k/freebase_mtr100_mte100-valid.txt"

opt_method="AdaGrad" # SGD, AdaGrad or AdaDelta
dimension="200"
margin="0.2"
step_size="0.01"
dist="0" # 1: L1 dist, 0: L2 dist

n="1"        # number of threads
n_mkl="-1"    # number of threads used by mkl. -1: automatically tuned
n_e="-1"     # number of threads for evaluation. -1: automatically tuned

./runTransE --n $n --n_mkl $n_mkl --n_e $n_e --opt $opt_method --d $dimension --margin $margin --step_size $step_size --dist $dist \
 --epoch 2000 --p_epoch 100 --o_epoch 100 --t_path $train_file --v_path $valid_file --e_path $test_file