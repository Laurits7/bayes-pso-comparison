#!/bin/bash

BASE=/home/user/Final_paper_results/HBC_BO_vs_PSO

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for i in {1..100}; do
    echo Beginning repeat "$i"/100
    python HBC_particle_swarm.py -o "$BASE"/PSO/"$i" -r "$i" &
    sleep 1
done

for i in {0..4}; do
    start=$(($i*20 + 1))
    stop=$((($i+1)*20))
    for ((j=$start;j<=$stop;j++)); do
        echo Beginning repeat "$j"/100
        python HBC_bayesian.py -o "$BASE"/Bayes/"$j" -r "$j" &
        sleep 4
    done
    sleep 86400
    echo New batch
done
