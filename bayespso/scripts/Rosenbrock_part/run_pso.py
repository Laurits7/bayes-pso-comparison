''' This script runs the Particle swarm optimization (PSO) with a given batch
size for 1000 repeats each optimization consists out of 10k total evaluations.
Call with 'python'

Usage: run_pso.py --batch_size=INT --output_dir=DIR

Options:
    -b --batch_size=INT             Path to parameters to be run
    -o --output_dir=DIR             Directory of the output
'''
import docopt
import os
import json
import numpy as np
import time
from bayespso.tools import particle_swarm as pso
from bayespso.tools import rosenbrock as rt

HYPERPARAMETER_INFO = {
    'x': {'max': 500., 'min':-500., 'int': 0, 'exp': 0, 'log': 0, 'power': 0},
    'y': {'max': 500., 'min':-500., 'int': 0, 'exp': 0, 'log': 0, 'power': 0}
}
TOTAL_EVALUATIONS = 10000
TOTAL_REPEATS = 1000


def main(batch_size, output_dir):
    iterations = int(TOTAL_EVALUATIONS / batch_size)
    start = time.time()
    output_dir = os.path.join(
        output_dir,
        'PSO/chunk_%s' %batch_size
    )
    pso_settings = {
        "iterations": iterations,
        "sample_size": batch_size,
        "compactness_threshold": 1e-3,
        "nr_informants": int(np.ceil(0.1 * batch_size)),
        "output_dir": output_dir,
        "seed": 1
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fitnesses = []
    for i in range(TOTAL_REPEATS):
        pso_settings['output_dir'] = os.path.join(output_dir, 'repeat_%s') %i
        np.random.seed(i)
        if not os.path.exists(pso_settings['output_dir']):
            os.makedirs(pso_settings['output_dir'])
        print('%s/%s' %(i, TOTAL_REPEATS))
        swarm = pso.ParticleSwarm(
            pso_settings, rt.ensemble_rosenbrock, HYPERPARAMETER_INFO)
        pso_best_parameters, pso_best_fitness = swarm.particleSwarmOptimization()
        fitnesses.append(pso_best_fitness)
    end = time.time()
    print("Total time: " + str(end-start))
    print("Mean: " + str(np.mean(fitnesses)))
    print("Std: " + str(np.std(fitnesses)))
    fitnesses_path = os.path.join(output_dir, 'fitnesses.txt')
    info_path = os.path.join(output_dir, 'info.json')
    with open(fitnesses_path, 'wt') as fitnesses_file:
        for fitness in fitnesses:
            fitnesses_file.write(str(fitness) + '\n')
    info = {
        'total_time': end-start,
        'fitness_mean': np.mean(fitnesses),
        'fitnesses_std': np.std(fitnesses),
        'chunk_size': batch_size,
        'total_repetitions': TOTAL_REPEATS,
        'total_evaluations': TOTAL_EVALUATIONS,
        'fitnesses_file': fitnesses_path
    }
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
