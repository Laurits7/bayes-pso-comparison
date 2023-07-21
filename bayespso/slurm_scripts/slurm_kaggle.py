
'''
Call with 'python3'

Usage: slurm_kaggle.py --parameter_file=PTH

Options:
    -p --parameter_file=PTH         Path to parameters to be run
'''
import docopt
import os
import json
from bayespso.tools import kaggle_score_calculator as ksc
from bayespso.tools import submission_higgs as sh


BASE = '/home/user/HBC_analysis/collected_evol_Kaggle/'
PATH_TO_TRAIN = '/home/user/training.csv'
PATH_TO_TEST = '/home/user/test.csv'
PATH_TO_TRUTH = '/home/user/atlas-higgs-challenge-2014-v2.csv'


def main(parameter_file):
    hyperparameters = read_parameters(parameter_file)
    repeat_nr = parameter_file.split('/')[-1].split('_')[0]
    submission_path = parameter_file.replace('json', 'submission')
    output_path = parameter_file.replace('.json', '_score.json')
    sh.submission_creation(
        PATH_TO_TRAIN,
        PATH_TO_TEST,
        hyperparameters,
        submission_path,
        seed=int(repeat_nr)
    )
    private, public = ksc.calculate_ams_scores(PATH_TO_TRUTH, submission_path)
    result = {
        'private': private,
        'public': public
    }
    with open(output_path, 'wt') as outFile:
        json.dump(result, outFile, indent=4)


def read_parameters(path):
    with open(path, 'rt') as inFile:
        parameters = json.load(inFile)
    return parameters



if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        parameter_file = arguments['--parameter_file']
        main(parameter_file)
    except docopt.DocoptExit as e:
        print(e)