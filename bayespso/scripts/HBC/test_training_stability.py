"""Call with 'python'. Stability with different seeds

Usage: bayes_vs_pso.py --output_dir=DIR

Options:
    -o --output_dir=DIR             Directory of the output
"""
import docopt
import os
import json
import numpy as np
import math
from bayespso.tools import kaggle_score_calculator as ksc
from bayespso.tools import submission_higgs as sh

PATH_TO_TRUTH = '/home/user/atlas-higgs-challenge-2014-v2.csv'
PATH_TO_TRAIN = '/home/user/training.csv'
PATH_TO_TEST = '/home/user/test.csv'
NUMBER_REPETITIONS = 100
HYPERPARAMETERS = {
    'PSO': {
         'colsample_bytree': 1.0,
         'gamma': 3.86,
         'learning_rate': 0.3,
         'max_depth': 4,
         'min_child_weight': 323.6,
         'num_boost_round': 153,
         'subsample': 0.830
    },
    'BO': {
         'colsample_bytree': 0.305766458,
         'gamma': 2.8856198,
         'learning_rate': 0.0480727149,
         'max_depth': 4,
         'min_child_weight': 208.063377,
         'num_boost_round': 462,
         'subsample': 0.894449083
    }
}


def main(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pso_outfile = os.path.join(output_dir, 'pso_submission.')
    bo_outfile = os.path.join(output_dir, 'bo_submission.')
    pso_results = {
        'private': [],
        'public': []
    }
    bo_results = {
        'private': [],
        'public': []
    }
    for i in range(NUMBER_REPETITIONS):
        sh.submission_creation(
            PATH_TO_TRAIN,
            PATH_TO_TEST,
            HYPERPARAMETERS['PSO'],
            pso_outfile + str(i),
            seed=i
        )
        sh.submission_creation(
            PATH_TO_TRAIN,
            PATH_TO_TEST,
            HYPERPARAMETERS['BO'],
            bo_outfile + str(i),
            seed=i
        )
        pso_private_ams, pso_public_ams = ksc.calculate_ams_scores(
            PATH_TO_TRUTH, pso_outfile + str(i)
        )
        bo_private_ams, bo_public_ams = ksc.calculate_ams_scores(
            PATH_TO_TRUTH, bo_outfile + str(i)
        )
        pso_results['public'].append(pso_public_ams)
        pso_results['private'].append(pso_private_ams)
        bo_results['public'].append(bo_public_ams)
        bo_results['private'].append(bo_private_ams)
    bo_file = os.path.join(output_dir, 'bo_results.json')
    pso_file = os.path.join(output_dir, 'pso_results.json')
    with open(bo_file, 'wt') as outFile:
        json.dump(bo_results, outFile)
    with open(pso_file, 'wt') as outFile:
        json.dump(pso_results, outFile)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        main(output_dir)
    except docopt.DocoptExit as e:
        print(e)