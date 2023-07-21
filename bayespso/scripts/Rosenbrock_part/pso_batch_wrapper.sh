#!/bin/bash

# This script is meant to parallelize the evaluation of different total eval

BASE=/home/user/PSO_batch_size_vol2
# TOTAL_EVAL=(100 250 500 1000 2500 5000 10000 15000)
TOTAL_EVAL=(350 600 850 1100 1350 1600 1850 2100 2350 2600 2850
       3100 3350 3600 3850 4100 4350 4600 4850 5100 5350 5600
       5850 6100 6350 6600 6850 7100 7350 7600 7850 8100 8350
       8600 8850 9100 9350 9600 9850)

for eval in ${TOTAL_EVAL[*]}; do
  python3 pso_batch_comparison.py -t "$eval" -o $BASE &
done

# Collect all results and plot?