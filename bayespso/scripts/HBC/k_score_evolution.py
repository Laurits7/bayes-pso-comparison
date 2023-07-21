import os
import json
from bayespso.tools import kaggle_score_calculator as ksc
from bayespso.tools import submission_higgs as sh
import glob
import shutil
import numpy as np
from textwrap import dedent
import time
import subprocess


BASE = '/home/user/HBC_analysis/collected_evol_Kaggle/'
PATH_TO_TRAIN = '/home/user/training.csv'
PATH_TO_TEST = '/home/user/test.csv'


def load_file(algorithm):
    input_file = os.path.join(BASE, '%s_results.json' %algorithm)
    with open(input_file, 'rt') as inFile:
        results = json.load(inFile)
    return results



def assign_to_results(results):
    wcp = os.path.join(BASE, 'tmp', '*', '*.json')
    for path in glob.glob(wcp):
        with open(path, 'rt') as inFile:
            scores = json.load(inFile)
        repeat_nr = path.split('/')[-2]
        iteration_nr = path.split('/')[-1].split('.')[0].split('_')[-1]
        for iteration_info in results[repeat_nr]:
            if iteration_info['iteration'] == int(iteration_nr):
                iteration_info.update(scores)



def wait_iteration(output_dir, total_repeats):
    '''Waits until all batch jobs are finised and in case of and warning
    or error that appears in the error file, stops running the optimization

    Parameters:
    ----------
    output_dir : str
        Path to the directory of output
    total_repeats : int
        Total of repetitions of the evaluation

    Returns:
    -------
    Nothing
    '''
    wild_card_path = os.path.join(BASE, 'tmp', '*', '*.json')
    while len(glob.glob(wild_card_path)) != total_repeats:
        check_error(output_dir)
        time.sleep(5)
        print('Still waiting ...')


def check_error(output_dir):
    '''In case of warnings or errors during batch job that is written to the
    error file, raises SystemExit(0)

    Parameters:
    ----------
    output_dir : str
        Path to the directory of the output, where the error file is located

    Returns:
    -------
    Nothing
    '''
    number_errors = 0
    error_list = ['FAILED', 'CANCELLED', 'ERROR', 'Error']
    output_error_list = ['Usage']
    error_files = os.path.join(output_dir, 'error*')
    output_files = os.path.join(output_dir, 'output*')
    for error_file in glob.glob(error_files):
        if os.path.exists(error_file):
            with open(error_file, 'rt') as file:
                lines = file.readlines()
                for line in lines:
                    for error in error_list:
                        if error in line:
                            number_errors += 1
    if number_errors > 0:
        print("Found errors: " + str(number_errors))
        raise SystemExit(0)



def prepare_job_file(
        parameter_file
):
    repeat_nr = parameter_file.split('/')[-5]
    iteration_nr = parameter_file.split('/')[-3].split('.')[0]
    output_dir = os.path.join(BASE, 'tmp', repeat_nr)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    job_file = os.path.join(
        output_dir, str(iteration_nr) + '.sh'
    )
    error_file = os.path.join(output_dir, 'error' + str(iteration_nr))
    output_file = os.path.join(output_dir, 'output' + str(iteration_nr))
    run_script = os.path.join(
        '/home/user/bayespso-paper-sw/bayespso',
        'slurm_scripts',
        'slurm_kaggle.py'
    )
    with open(job_file, 'wt') as filehandle:
        filehandle.writelines(dedent(
            """
                #!/bin/bash
                #SBATCH --job-name=kaggleScore
                #SBATCH --partition=short
                #SBATCH --ntasks=1
                #SBATCH --time=02:00:00
                #SBATCH --cpus-per-task=4
                #SBATCH -e %s
                #SBATCH -o %s
                env
                date
                python %s -p %s
            """ % (
                    error_file, output_file, run_script, parameter_file
            )
        ).strip('\n'))
    return job_file



def main(algorithm):
    results = load_file(algorithm)
    total_repeats = 0
    for repeat in results.keys():
        repeat_info = results[repeat]
        for iteration_info in repeat_info:
            total_repeats += 1
            job_file = prepare_job_file(iteration_info['path'])
            subprocess.call(['sbatch', job_file])
    output_dir = os.path.join(BASE, 'tmp', '*')
    wait_iteration(output_dir, total_repeats)
    time.sleep(30)
    assign_to_results(results)
    output_path = os.path.join(BASE, 'updated_results_%s.json' %algorithm)
    with open(output_path, 'wt') as outFile:
        json.dump(results, outFile, indent=4)


if __name__ == '__main__':
    # main('PSO')
    main('Bayes')
