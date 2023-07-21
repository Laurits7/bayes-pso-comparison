'''
Call with 'python3'

Usage: batch_evaluation.py --batch_size=INT --output_dir=DIR --total_evals=INT

Options:
    -b --batch_size=INT             Batch size
    -t --total_evals=INT            Number of total evaluations done by PSO
    -o --output_dir=DIR             Directory of the output
'''
import docopt
import os
import json
import numpy as np
from bayespso.tools import particle_swarm as pso
from bayespso.tools import rosenbrock as rt


VALUE_DICTS = {
    'x': {'max': 500., 'min':-500., 'int': 0, 'exp': 0, 'log': 0, 'power': 0},
    'y': {'max': 500., 'min':-500., 'int': 0, 'exp': 0, 'log': 0, 'power': 0}
}
TOTAL_REPEATS = 100

def main(total_eval, batch_size, output_dir):
    iterations = int(np.ceil(total_eval/batch_size))
    os.makedirs(output_dir, exist_ok=True)
    pso_settings = {
        "iterations": iterations,
        "sample_size": batch_size,
        "compactness_threshold": 1e-3,
        "nr_informants": int(np.ceil(0.1 * batch_size)),
        "output_dir": output_dir,
        "seed": 1,
        "continue": 0
    }
    fitnesses = []
    for i in range(TOTAL_REPEATS):
        pso_settings['output_dir'] = os.path.join(output_dir, 'repeat_%s') %i
        np.random.seed(i)
        if not os.path.exists(pso_settings['output_dir']):
            os.makedirs(pso_settings['output_dir'])
        print('------------ Repeat nr %s/%s ----- Total eval: %s \t Batch size: %s--------' %(
            i, TOTAL_REPEATS, total_eval, batch_size))
        swarm = pso.ParticleSwarm(
            pso_settings, rt.ensemble_rosenbrock, VALUE_DICTS)
        pso_best_parameters, pso_best_fitness = swarm.particleSwarmOptimization()
        fitnesses.append((-1)*pso_best_fitness)
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


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        TOTAL_EVAL = int(arguments['--total_evals'])
        BATCH_SIZE = int(arguments['--batch_size'])
        OUTPUT_DIR = arguments['--output_dir']
        main(TOTAL_EVAL, BATCH_SIZE, OUTPUT_DIR)
    except docopt.DocoptExit as e:
        print(e)