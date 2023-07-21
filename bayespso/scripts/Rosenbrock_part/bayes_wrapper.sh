#!/bin/bash

BASE=/home/user/Bayes_PSO
BATCH_SIZES=(1 2 5 10 25 50 100 250 500 1000)
#BATCH_SIZES_EVEN=(2 10 250 1000)
# BATCH_SIZES_ODD=(1 5 25 50 100 500)

for size in ${BATCH_SIZES[*]}; do
  python3 run_bayesian_opt.py -b "$size" -o $BASE
done
