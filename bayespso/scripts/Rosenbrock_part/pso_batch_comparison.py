''' The purpose of this script is to analyze, whether the optimal number of
particles for a given max_eval is 1% or sqrt(max_eval)
Eval points gotten with:

curr_eval = 10000
test_points = [100, 1000, 10000, 100000, 1000000]
single_test_internal = np.array([1, 5, 25, 50, 100, 500, 1000, 2000, 5000])
fractions = np.array([i/curr_eval for i in single_test_internal])
points = {}
for i in test_points:
    evals = fractions*i
    points[i] = sorted(list(set([int(np.ceil(j)) for j in evals])))

Usage: pso_batch_comparison.py --total_eval=INT --output_dir=DIR

Options:
    -t --total_eval=INT             Total number of evaluations to run
    -o --output_dir=DIR             Directory of the output
'''
import docopt
import os
import json
import numpy as np
import time
from bayespso.tools import particle_swarm as pso
from bayespso.tools import rosenbrock as rt


TOTAL_REPEATS = 100
# totalEvals = [100, 250, 500, 1000, 2500, 5000, 10000, 15000]
# totalEvals = [ 350,  600,  850, 1100, 1350, 1600, 1850, 2100, 2350, 2600, 2850,
#        3100, 3350, 3600, 3850, 4100, 4350, 4600, 4850, 5100, 5350, 5600,
#        5850, 6100, 6350, 6600, 6850, 7100, 7350, 7600, 7850, 8100, 8350,
#        8600, 8850, 9100, 9350, 9600, 9850]
totalEvals = [3000]
# nr_eval_points = 50
EVAL_POINTS = {}
# for evalPoint in totalEvals:
#     evals = np.linspace(start=1, stop=evalPoint, num=nr_eval_points)
#     evals = list(sorted([int(np.ceil(ev)) for ev in evals]))
#     EVAL_POINTS[evalPoint] = evals

for evalPoint in totalEvals:
    BATCH_SIZES = []
    for i in range(1, evalPoint + 1):
        if evalPoint % i == 0:
            BATCH_SIZES.append(i)
    EVAL_POINTS[evalPoint] = BATCH_SIZES
HYPERPARAMETER_INFO = {
    'x': {'max': 500., 'min':-500., 'int': 0, 'exp': 0, 'log': 0, 'power': 0},
    'y': {'max': 500., 'min':-500., 'int': 0, 'exp': 0, 'log': 0, 'power': 0}
}


def run_for_a_given_batch_size(batch_size, total_eval, output_dir):
    print("Total repeats: %s" %total_eval)
    iterations = int(total_eval / batch_size)
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
        "seed": 42
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fitnesses = []
    for i in range(TOTAL_REPEATS):
        pso_settings['output_dir'] = os.path.join(output_dir, 'repeat_%s') %i
        np.random.seed(i)
        if not os.path.exists(pso_settings['output_dir']):
            os.makedirs(pso_settings['output_dir'])
        print('------------ Repeat nr %s/%s ----- Total eval: %s \t Batch size: %s--------' %(
            i, TOTAL_REPEATS, total_eval, batch_size))
        swarm = pso.ParticleSwarm(
            pso_settings, rt.ensemble_rosenbrock, HYPERPARAMETER_INFO)
        pso_best_parameters, pso_best_fitness = swarm.particleSwarmOptimization()
        fitnesses.append((-1)*pso_best_fitness)
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
        'total_evaluations': total_eval,
        'fitnesses_file': fitnesses_path
    }
    with open(info_path, 'wt') as info_file:
        json.dump(info, info_file)


def main(total_eval, main_dir):
    output_dir = os.path.join(main_dir, 'total_eval_%s' %total_eval)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for batch_size in EVAL_POINTS[total_eval]:
        run_for_a_given_batch_size(batch_size, total_eval, output_dir)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        TOTAL_EVAL = int(arguments['--total_eval'])
        OUTPUT_DIR = arguments['--output_dir']
        main(TOTAL_EVAL, OUTPUT_DIR)
    except docopt.DocoptExit as e:
        print(e)