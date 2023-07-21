import time
from pathlib import Path
import os
import subprocess
import json
import csv
import glob
import shutil
import numpy as np
from textwrap import dedent

THREADS_AVAILABLE = 4
MAX_JOBS = 5000

def repetition_evaluation(
        batch_size,
        output_dir,
        total_evaluations,
        total_repeats
):
    """The main function call that is the slurm equivalent of ensemble_fitness
    in xgb_tools

    Parameters:
    ----------
    batch_size : int
        Number of evaluations to be done in parallel
    output_dir : str
        Directory of the main output.
    total_evaluations : int
        Number of evaluations to be done in total
    total_repeats : int
        Total number of repats to be made in order to get a better estimate
        of the mean performance.

    Returns:
    -------
    scores : list of floats
        Fitnesses for each hyperparameter-set
    """
    for repeat in range(total_repeats):
        job_file = prepare_job_file(
            batch_size, output_dir, total_evaluations, repeat
        )
        subprocess.call(['sbatch', job_file])
    wait_iteration(output_dir, total_repeats)
    time.sleep(30)
    scores = read_fitness(output_dir)
    return scores


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
    wild_card_path = os.path.join(output_dir, '*', 'result.txt')
    while len(glob.glob(wild_card_path)) != total_repeats:
        check_error(output_dir)
        time.sleep(5)


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


def read_fitness(output_dir):
    '''Creates the list of score dictionaries of each sample. List is ordered
    according to the number of the sample

    Parameters:
    ----------
    output_dir : str
        Path to the directory of output

    Returns:
    -------
    scores : list of floats
        List of fitnesses
    '''
    scores = []
    wild_card_path = os.path.join(output_dir, '*', 'result.txt')
    for path in glob.glob(wild_card_path):
        with open(path, 'rt') as result_file:
            for line in result_file:
                scores.append(float(line.strip('\n')))
    return scores


def prepare_job_file(
        batch_size,
        output_dir,
        total_eval,
        repeat_nr
):
    """Writes the job file that will be executed by slurm

    Parameters:
    ----------
    parameter_file : str
        Path to the parameter file
    sample_nr : int
        Number of the sample (parameter-set)
    global_settings : dict
        Global settings for the run

    Returns:
    -------
    job_file : str
        Path to the script to be executed by slurm
    """
    main_dir = find_package_dir()
    repeat_dir = os.path.join(output_dir, str(repeat_nr))
    job_dir = os.path.join(
        output_dir, 'job_dir'
    )
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
    if not os.path.exists(repeat_dir):
        os.makedirs(repeat_dir)
    job_file = os.path.join(
        job_dir, 'repeat_' + str(repeat_nr) + '.sh'
    )
    error_file = os.path.join(repeat_dir, 'error' + str(repeat_nr))
    output_file = os.path.join(repeat_dir, 'output' + str(repeat_nr))
    run_script = os.path.join(
        main_dir,
        'slurm_scripts',
        'slurm_bayesian.py'
    )
    with open(job_file, 'wt') as filehandle:
        filehandle.writelines(dedent(
            """
                #!/bin/bash
                #SBATCH --job-name=BO_7k_mem
                #SBATCH --partition=mem
                #SBATCH --ntasks=1
                #SBATCH --time=7-00:00:00
                #SBATCH --mem-per-cpu=3500mb
                #SBATCH --cpus-per-task=%(nthread)s
                #SBATCH -e %(error_file)s
                #SBATCH -o %(output_file)s
                env
                date
                export OMP_NUM_THREADS=%(nthread)s
                export OPENBLAS_NUM_THREADS=%(nthread)s
                export MKL_NUM_THREADS=%(nthread)s
                export VECLIB_MAXIMUM_THREADS=%(nthread)s
                export NUMEXPR_NUM_THREADS=%(nthread)s
                python %(run_script)s -b %(batch_size)s -o %(repeat_dir)s -e %(total_eval)s -r %(repeat_nr)s
            """ % {
                'nthread': 2,
                'error_file': error_file,
                'output_file': output_file,
                'run_script': run_script,
                'batch_size': batch_size,
                'repeat_dir': repeat_dir,
                'total_eval': total_eval,
                'repeat_nr': repeat_nr
            }
        ).strip('\n'))
    return job_file


def find_package_dir():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(dir_path, os.pardir))




def repetition_evaluation_skopt(
        batch_size,
        output_dir,
        total_evaluations,
        total_repeats
):
    """The main function call that is the slurm equivalent of ensemble_fitness
    in xgb_tools

    Parameters:
    ----------
    batch_size : int
        Number of evaluations to be done in parallel
    output_dir : str
        Directory of the main output.
    total_evaluations : int
        Number of evaluations to be done in total
    total_repeats : int
        Total number of repats to be made in order to get a better estimate
        of the mean performance.

    Returns:
    -------
    scores : list of floats
        Fitnesses for each hyperparameter-set
    """
    for repeat in range(total_repeats):
        job_file = prepare_job_file_skopt(
            batch_size, output_dir, total_evaluations, repeat
        )
        subprocess.call(['sbatch', job_file])
    wait_iteration(output_dir, total_repeats)
    time.sleep(30)
    scores = read_fitness(output_dir)
    return scores


def prepare_job_file_skopt(
        batch_size,
        output_dir,
        total_eval,
        repeat_nr
):
    """Writes the job file that will be executed by slurm

    Parameters:
    ----------
    parameter_file : str
        Path to the parameter file
    sample_nr : int
        Number of the sample (parameter-set)
    global_settings : dict
        Global settings for the run

    Returns:
    -------
    job_file : str
        Path to the script to be executed by slurm
    """
    main_dir = find_package_dir()
    repeat_dir = os.path.join(output_dir, str(repeat_nr))
    job_dir = os.path.join(
        output_dir, 'job_dir'
    )
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
    if not os.path.exists(repeat_dir):
        os.makedirs(repeat_dir)
    job_file = os.path.join(
        job_dir, 'repeat_' + str(repeat_nr) + '.sh'
    )
    error_file = os.path.join(repeat_dir, 'error' + str(repeat_nr))
    output_file = os.path.join(repeat_dir, 'output' + str(repeat_nr))
    run_script = os.path.join(
        main_dir,
        'slurm_scripts',
        'slurm_skopt_bayesian.py'
    )
    with open(job_file, 'wt') as filehandle:
        filehandle.writelines(dedent(
            """
                #!/bin/bash
                #SBATCH --job-name=bayesOpt
                #SBATCH --partition=main
                #SBATCH --ntasks=1
                #SBATCH --time=2-00:00:00
                #SBATCH --cpus-per-task=4
                #SBATCH -e %s
                #SBATCH -o %s
                env
                date
                python %s -b %s -o %s -e %s -r %s
            """ % (
                    error_file, output_file, run_script, batch_size,
                    repeat_dir, total_eval, repeat_nr
            )
        ).strip('\n'))
    return job_file



###########################


def repetition_branin_skopt(
        batch_size,
        output_dir,
        total_evaluations,
        total_repeats
):
    """The main function call that is the slurm equivalent of ensemble_fitness
    in xgb_tools

    Parameters:
    ----------
    batch_size : int
        Number of evaluations to be done in parallel
    output_dir : str
        Directory of the main output.
    total_evaluations : int
        Number of evaluations to be done in total
    total_repeats : int
        Total number of repats to be made in order to get a better estimate
        of the mean performance.

    Returns:
    -------
    scores : list of floats
        Fitnesses for each hyperparameter-set
    """
    for repeat in range(total_repeats):
        job_file = prepare_job_file_branin_skopt(
            batch_size, output_dir, total_evaluations, repeat
        )
        subprocess.call(['sbatch', job_file])
    wait_iteration(output_dir, total_repeats)
    time.sleep(30)
    scores = read_fitness(output_dir)
    return scores


def prepare_job_file_branin_skopt(
        batch_size,
        output_dir,
        total_eval,
        repeat_nr
):
    """Writes the job file that will be executed by slurm

    Parameters:
    ----------
    parameter_file : str
        Path to the parameter file
    sample_nr : int
        Number of the sample (parameter-set)
    global_settings : dict
        Global settings for the run

    Returns:
    -------
    job_file : str
        Path to the script to be executed by slurm
    """
    main_dir = find_package_dir()
    repeat_dir = os.path.join(output_dir, str(repeat_nr))
    job_dir = os.path.join(
        output_dir, 'job_dir'
    )
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
    if not os.path.exists(repeat_dir):
        os.makedirs(repeat_dir)
    job_file = os.path.join(
        job_dir, 'repeat_' + str(repeat_nr) + '.sh'
    )
    error_file = os.path.join(repeat_dir, 'error' + str(repeat_nr))
    output_file = os.path.join(repeat_dir, 'output' + str(repeat_nr))
    run_script = os.path.join(
        main_dir,
        'slurm_scripts',
        'slurm_skopt_branin_bayesian.py'
    )
    with open(job_file, 'wt') as filehandle:
        filehandle.writelines(dedent(
            """
                #!/bin/bash
                #SBATCH --job-name=bayesOpt
                #SBATCH --partition=main
                #SBATCH --ntasks=1
                #SBATCH --time=2-00:00:00
                #SBATCH --cpus-per-task=4
                #SBATCH -e %s
                #SBATCH -o %s
                env
                date
                python %s -b %s -o %s -e %s -r %s
            """ % (
                    error_file, output_file, run_script, batch_size,
                    repeat_dir, total_eval, repeat_nr
            )
        ).strip('\n'))
    return job_file