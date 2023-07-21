#!/bin/bash

BASE=/home/user/Bayes_PSO
BATCH_SIZES=(1 2 5 10 25 50 100 250 500 1000)

for size in ${BATCH_SIZES[*]}; do
  python3 run_pso.py -b "$size" -o $BASE
done

python3 collect_pso_results.py -i "$BASE" -o "$BASE"
