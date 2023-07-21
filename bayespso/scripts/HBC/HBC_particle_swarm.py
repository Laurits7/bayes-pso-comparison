'''
Call with 'python'

Usage: higgs_pso_xgb_pso.py --output_dir=DIR --repeat=INT

Options:
    -o --output_dir=DIR     Directory of the output
    -r --repeat=INT         Number of the repeat
'''
from bayespso.tools import slurm_tools as st
from bayespso.tools import submission_higgs as sh
from bayespso.tools import particle_swarm as pm
from pathlib import Path
import numpy as np
import shutil
import os
import json
import docopt

np.random.seed(1)
data_path = '/home/user'

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


pso_settings = {
    "iterations": 100,
    "sample_size": 70,
    "nr_informants": 7
}


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
    pso_settings.update(global_settings)
    swarm = pm.ParticleSwarm(pso_settings, st.get_fitness_score, HYPERPARAMETERS)
    optimal_hyperparameters = swarm.particleSwarmOptimization()[0]
    save_results(optimal_hyperparameters, output_dir)



def save_results(optimal_hyperparameters, output_dir):
    path_to_test = os.path.join(data_path, 'test.csv')
    path_to_train = os.path.join(data_path, 'training.csv')
    score_path = os.path.join(output_dir, 'best_hyperparameters.json')
    outfile = os.path.join(output_dir, 'higgs_submission.pso')
    with open(score_path, 'w') as out_file:
        json.dump(optimal_hyperparameters, out_file)
    print('Creating submission file')
    sh.submission_creation(
        path_to_train,
        path_to_test,
        optimal_hyperparameters,
        outfile
    )
    print('Results saved to ' + str(output_dir))
    print('PSO submission file: ' + str(outfile))


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        repeat = arguments['--repeat']
        main(output_dir, repeat)
    except docopt.DocoptExit as e:
        print(e)

