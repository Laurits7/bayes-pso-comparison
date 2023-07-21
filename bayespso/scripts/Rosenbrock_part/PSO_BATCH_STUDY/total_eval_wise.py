'''
Call with 'python3'

Usage: slurm_xgb_atlas.py --output_dir=DIR --total_evals=INT

Options:
    -o --output_dir=DIR             Directory of the output
    -t --total_evals=INT            Number of total evaluations done by PSO
'''
import docopt
import os
from textwrap import dedent
import numpy as np
import subprocess


# BATCH_SIZES = [int(i) for i in np.linspace(start=1, stop=350, num=50)]


def main(total_evals, output_dir):
    BATCH_SIZES = []
    for i in range(1, total_evals + 1):
        if total_evals % i == 0:
            BATCH_SIZES.append(i)
    totalEval_outputDir = os.path.join(output_dir, str(total_evals))
    os.makedirs(totalEval_outputDir, exist_ok=True)
    for batch_size in BATCH_SIZES:
        batch_outputDir = os.path.join(totalEval_outputDir, 'batch_%s' %batch_size)
        os.makedirs(batch_outputDir, exist_ok=True)
        job_file_path = create_job_file(total_evals, batch_size, batch_outputDir)
        subprocess.call(['sbatch', job_file_path])


def create_job_file(total_evals, batch_size, batch_outputDir):
    main_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.join(main_dir, 'batch_evaluation.py')
    error_folder = os.path.join(batch_outputDir, 'errors')
    os.makedirs(error_folder, exist_ok=True)
    error_file = os.path.join(error_folder, 'error' + str(batch_size))
    output_folder = os.path.join(batch_outputDir, 'outputs')
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'output' + str(batch_size))
    job_folder = os.path.join(batch_outputDir, 'jobs')
    os.makedirs(job_folder, exist_ok=True)
    job_file = os.path.join(job_folder, 'batch_submission_%s.sh' %batch_size)
    results_folder = os.path.join(batch_outputDir, 'results')
    os.makedirs(results_folder, exist_ok=True)
    with open(job_file, 'wt') as filehandle:
        filehandle.writelines(dedent(
            """
                #!/bin/bash
                #SBATCH --job-name=CHUNK
                #SBATCH --ntasks=1
                #SBATCH -p short
                #SBATCH --time=2:00:00
                #SBATCH --cpus-per-task=1
                #SBATCH -e %s
                #SBATCH -o %s
                env
                date
                python3 %s  -b %s -t %s -o %s
            """ % (
                    error_file, output_file,
                    script_name, batch_size, total_evals, results_folder
            )
        ).strip('\n'))
    return job_file


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        TOTAL_EVALS = int(arguments['--total_evals'])
        OUTPUT_DIR = arguments['--output_dir']
        main(TOTAL_EVALS, OUTPUT_DIR)
    except docopt.DocoptExit as e:
        print(e)