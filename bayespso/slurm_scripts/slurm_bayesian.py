'''
Call with 'python'

Usage: slurm_bayesian.py --batch_size=INT --output_dir=DIR --total_eval=INT --repeat=INT

Options:
    -b --batch_size=INT             Path to parameters to be run
    -o --output_dir=DIR             Directory of the output
    -e --total_eval=INT             Total number of evaluations
    -r --repeat=INT                 Number of the repeat
'''
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import numpy as np
import docopt
import json
from pathlib import Path
import time
from bayespso.tools import bayesian_opt as bo
from bayespso.tools import rosenbrock as rt

HYPERPARAMETERS = {
    'x': {'max': 500., 'min':-500., 'int': 0, 'exp': 0, 'log': 0, 'power': 0},
    'y': {'max': 500., 'min':-500., 'int': 0, 'exp': 0, 'log': 0, 'power': 0}
}


def main(batch_size, total_eval, output_dir, repeat):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.random.seed(repeat)
    global_settings = {
        'output_dir': output_dir,
        'seed': repeat
    }
    iterations = int(total_eval / batch_size)
    bayesian_optimizer = bo.BayesianOptimizer(
        HYPERPARAMETERS,
        rt.ensemble_rosenbrock,
        global_settings,
        niter=iterations,
        nparallel_eval=batch_size
    )
    minimum = float(bayesian_optimizer.optimize()[1])
    outfile = os.path.join(output_dir, 'result.txt')
    with open(outfile, 'wt') as out_file:
        out_file.write(str(minimum))


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        batch_size = int(arguments['--batch_size'])
        total_eval = int(arguments['--total_eval'])
        repeat = int(arguments['--repeat'])
        output_dir = arguments['--output_dir']
        main(batch_size, total_eval, output_dir, repeat)
    except docopt.DocoptExit as e:
        print(e)
