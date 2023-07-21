#!/bin/bash

BASE=/home/user/Final_paper_results/Ros_7k_BO_vs_PSO


export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SIZE=100


python3 run_bayesian_opt.py -b "$SIZE" -o $BASE &
python3 run_pso.py -b "$SIZE" -o $BASE