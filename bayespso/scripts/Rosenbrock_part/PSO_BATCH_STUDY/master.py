''' 

Usage: pso_batch_comparison.py --output_dir=DIR

Options:
    -o --output_dir=DIR             Directory of the output
'''

import docopt
import os
import json
import numpy as np
import time
from bayespso.tools import particle_swarm as pso
from bayespso.tools import rosenbrock as rt
from textwrap import dedent
import subprocess


TOTAL_EVALS = [int(nr_evals) for nr_evals in np.linspace(100, 10000, num=50)]


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for nr_evals in TOTAL_EVALS:
        totalEval_outputDir = os.path.join(output_dir, str(nr_evals))
        os.makedirs(totalEval_outputDir, exist_ok=True)
        job_file_path = create_job_file(nr_evals, totalEval_outputDir)
        subprocess.call(['sbatch', job_file_path])


def create_job_file(total_evals, totalEval_outputDir):
    main_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.join(main_dir, 'total_eval_wise.py')
    error_folder = os.path.join(totalEval_outputDir, 'errors')
    os.makedirs(error_folder, exist_ok=True)
    error_file = os.path.join(error_folder, 'error' + str(total_evals))
    output_folder = os.path.join(totalEval_outputDir, 'outputs')
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'output' + str(total_evals))
    job_folder = os.path.join(totalEval_outputDir, 'jobs')
    os.makedirs(job_folder, exist_ok=True)
    job_file = os.path.join(job_folder, 'batch_submission.sh')
    results_folder = os.path.join(totalEval_outputDir, 'results')
    os.makedirs(results_folder, exist_ok=True)
    with open(job_file, 'wt') as filehandle:
        filehandle.writelines(dedent(
            """
                #!/bin/bash
                #SBATCH --job-name=TOTAL_E
                #SBATCH --ntasks=1
                #SBATCH -p short
                #SBATCH --time=2:00:00
                #SBATCH --cpus-per-task=1
                #SBATCH -e %s
                #SBATCH -o %s
                env
                date
                python3 %s -t %s -o %s
            """ % (
                    error_file, output_file,
                    script_name, total_evals, results_folder
            )
        ).strip('\n'))
    return job_file






if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        OUTPUT_DIR = arguments['--output_dir']
        main(OUTPUT_DIR)
    except docopt.DocoptExit as e:
        print(e)