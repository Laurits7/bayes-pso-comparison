''' This script runs the Particle swarm optimization (PSO) with a given batch
size for 1000 repeats each optimization consists out of 10k total evaluations.
Call with 'python'

Usage: run_pso.py --batch_size=INT --output_dir=DIR

Options:
    -b --batch_size=INT             Path to parameters to be run
    -o --output_dir=DIR             Directory of the output
'''
import os
import json
import numpy as np
import time
import docopt
from bayespso.tools import slurm_helper as sh
from bayespso.tools import bayesian_opt as bo

HYPERPARAMETER_INFO = {
    'x': {'max': 500., 'min':-500., 'int': 0, 'exp': 0, 'log': 0, 'power': 0},
    'y': {'max': 500., 'min':-500., 'int': 0, 'exp': 0, 'log': 0, 'power': 0}
}
TOTAL_EVALUATIONS = 7000
TOTAL_REPEATS = 1000


# If PSO vs BO @ 100 parallel then change TOTAL_REPEATS to 1000 and the partition to long

def main(batch_size, output_dir):
    batch_output_dir = os.path.join(
        output_dir, 'BayesianOpt', 'batch_%s' % batch_size
    )
    if not os.path.exists(batch_output_dir):
        os.makedirs(batch_output_dir)
    start = time.time()
    fitnesses = sh.repetition_evaluation(
        batch_size, batch_output_dir, TOTAL_EVALUATIONS, TOTAL_REPEATS
    )
    end = time.time()
    print("Total time: " + str(end-start))
    print("Mean: " + str(np.mean(fitnesses)))
    print("Std: " + str(np.std(fitnesses)))
    info = {
        'total_time': end-start,
        'fitness_mean': np.mean(fitnesses),
        'fitnesses_std': np.std(fitnesses),
        'chunk_size': batch_size,
        'total_repetitions': TOTAL_REPEATS,
        'total_evaluations': TOTAL_EVALUATIONS
    }
    info_path = os.path.join(output_dir, 'info.json')
    with open(info_path, 'wt') as info_file:
        json.dump(info, info_file)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        batch_size = int(arguments['--batch_size'])
        output_dir = arguments['--output_dir']
        main(batch_size, output_dir)
    except docopt.DocoptExit as e:
        print(e)
