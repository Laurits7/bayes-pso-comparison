''' This script runs particle swarm optimization for the HBC.
Call with 'python'

Usage: HBC_bayesian.py --output_dir=DIR --repeat=INT

Options:
    -o --output_dir=INT     Directory of the output
    -r --repeat=INT         Number of the repeat
'''
import os
import json
import numpy as np
import time
import docopt
from bayespso.tools import slurm_tools as st
from bayespso.tools import bayesian_opt as bo
from bayespso.tools import submission_higgs as sh


HYPERPARAMETERS = {
    "num_boost_round": {
        "min": 1,
        "max": 500,
        "int": 1,
        "exp": 0,
        "log": 0,
        "power": 0
    },
    "learning_rate": {
        "min": -5.,
        "max": 0.,
        "int": 0,
        "exp": 1,
        "log": 0,
        "power": 0
    },
    "max_depth": {
        "min": 1.,
        "max": 6.,
        "int": 1,
        "exp": 0,
        "log": 0,
        "power": 0
    },
    "gamma": {
        "min": 0.,
        "max": 5.,
        "int": 0,
        "exp": 0,
        "log": 0,
        "power": 0
    },
    "min_child_weight": {
        "min": 0.,
        "max": 500.,
        "int": 0,
        "exp": 0,
        "log": 0,
        "power": 0
    },
    "subsample": {
        "min": 0.8,
        "max": 1,
        "int": 0,
        "exp": 0,
        "log": 0,
        "power": 0
    },
    "colsample_bytree": {
        "min": 0.3,
        "max": 1,
        "int": 0,
        "exp": 0,
        "log": 0,
        "power": 0
    }
}

data_path = '/home/user'


def main(output_dir, repeat):
    global_settings = {
        'output_dir': output_dir,
        'nthread': 4,
        'kappa': 0.3,
        'package_dir': '/home/user/bayespso-paper-sw/bayespso',
        'seed': int(repeat),
        'continue': 1
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    bayesian_optimizer = bo.BayesianOptimizer(
        HYPERPARAMETERS,
        st.get_fitness_score,
        global_settings,
        niter=100,
        nparallel_eval=70
    )
    minimum_location = bayesian_optimizer.optimize()[0]
    result_dict = list_to_dict(minimum_location)
    print('Found minima found at: ' + str(minimum_location))
    save_results(result_dict, data_path, output_dir)


def list_to_dict(minimum_location):
    result_dict = {}
    for location, hyperparameter in zip(minimum_location, HYPERPARAMETERS):
        if HYPERPARAMETERS[hyperparameter]['int'] == 1:
            location = int(location)
        result_dict[hyperparameter] = location
    return result_dict



def save_results(optimal_hyperparameters, data_path, output_dir):
    path_to_test = os.path.join(data_path, 'test.csv')
    path_to_train = os.path.join(data_path, 'training.csv')
    score_path = os.path.join(output_dir, 'best_hyperparameters.json')
    outfile = os.path.join(output_dir, 'higgs_submission.pso')
    with open(score_path, 'w') as file:
        json.dump(optimal_hyperparameters, file)
    print('Creating submission file')
    sh.submission_creation(
        path_to_train,
        path_to_test,
        optimal_hyperparameters,
        outfile
    )
    print('Results saved to ' + str(output_dir))
    print('Bayes submission file: ' + str(outfile))


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        repeat = arguments['--repeat']
        main(output_dir, repeat)
    except docopt.DocoptExit as e:
        print(e)