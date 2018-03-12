#!/usr/bin/env bash

rm -rf build-rt
mkdir build-rt
cd build-rt
cmake ../../../../
make

#train_file="../../../../data/WN18/wordnet-mlj12-train.txt"
#test_file="../../../../data/WN18/wordnet-mlj12-test.txt"
#valid_file="../../../../data/WN18/wordnet-mlj12-valid.txt"

train_file="../../../../data/FB15k/freebase_mtr100_mte100-train.txt"
test_file="../../../../data/FB15k/freebase_mtr100_mte100-test.txt"
valid_file="../../../../data/FB15k/freebase_mtr100_mte100-valid.txt"

n="-1"        # number of threads
n_mkl="-1"    # number of threads used by mkl. -1: automatically tuned
n_e="-1"     # number of threads for evaluation. -1: automatically tuned

rescal_d="200"
transe_d="200"
rescal_r_epoch="2000"
transe_r_epoch="2000"

rescal_path="../../../../../output/FB15K/RESCALRANK-200-62/output/2000"
transe_path="../../../../../output/FB15K/TransE-200-80/output/2000"

dist="1" # 1: L1 dist, 0: L2 dist

r_t="100" # duplicate 2 * r_t true triples, r_t faked triples for replacing subjects and r_t faked triples for replacing objects

c="1"        # parameter for liblinear

# cannot be both 1
normalize="1" # normalize value before logistic regression
znormalize="0" # normalize using zscore

./runRTLREnsemble --c $c --normalize $normalize --znormalize $znormalize --n $n --n_mkl $n_mkl --n_e $n_e --r_t $r_t \
--rescal_d $rescal_d --transe_d $transe_d --rescal_r_epoch $rescal_r_epoch --transe_r_epoch $transe_r_epoch \
--rescal_path $rescal_path --transe_path $transe_path --dist $dist --t_path $train_file --v_path $valid_file --e_path $test_file
