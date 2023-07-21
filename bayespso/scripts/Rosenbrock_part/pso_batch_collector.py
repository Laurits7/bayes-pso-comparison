''' This script is meant for collecting the results of the batch_size vs.
total evaluation study for the PSO '''

import numpy as np
import glob
import os
import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Note:
#    - SQRT scenario -> #Iter = #Particles
#   - 1% scenario -> #Iter = 100


STUDY_DIR = '/home/user/PSO_batch_size_vol2'
RESULT_DIR = '/home/user/PSO_batch_results2'

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def collect_totalEval_scenario():
    totalEval_info = {}
    totalEval_scenario_wcp = os.path.join(STUDY_DIR, 'total_eval_*')
    for totalEval_dir in glob.glob(totalEval_scenario_wcp):
        total_eval = int(os.path.basename(totalEval_dir).split('_')[-1])
        totalEval_info[total_eval] = {}
        batch_size_infos = os.path.join(totalEval_dir, 'PSO/chunk_*/info.json')
        for chunk_info_path in glob.glob(batch_size_infos):
            chunk_size = int(chunk_info_path.split('/')[-2].split('_')[-1])
            info = read_json(chunk_info_path)
            totalEval_info[total_eval][chunk_size] = info
    return totalEval_info


def read_json(path):
    with open(path, 'rt') as inFile:
        info = json.load(inFile)
    return info


def print_info(totalEvals, sqrt_results, one_percent_results, optimal_points):
    print('1% hypo \t | sqrt hypo \t \t| Total evaluations \t | Optimal')
    print('-----------------------------------------------------------')
    for i, j, k, l in zip(one_percent_results, sqrt_results, totalEvals, optimal_points):
        print('%s \t \t | %s \t \t | %s \t | %s' %(i, round(j, 2), k, l))



def plot_optimal_values(optimal_points, totalEvals):
    totalEvals = sorted(totalEvals)
    sqrt_results = [np.sqrt(evalSize) for evalSize in totalEvals]
    one_percent_results = [evalSize/100 for evalSize in totalEvals]
    print_info(totalEvals, sqrt_results, one_percent_results, optimal_points)
    plt.plot(
        totalEvals, optimal_points, color='k', marker='',
        label='True optimal points'
    )
    plt.plot(totalEvals, sqrt_results, color='k', marker='^',
        label='Sqrt hypothesis', ls=''
    )
    plt.plot(totalEvals, one_percent_results, color='k', marker='s',
        label='1% hypothesis', ls=''
    )
    plt.grid()
    plt.xlabel('# total evaluations')
    plt.xscale('log')
    plt.ylabel('Optimal batch size')
    plt.legend(loc='upper left')
    plt.savefig(
        os.path.join(RESULT_DIR, 'hypothesis.png'), bbox_inches='tight'
    )
    plt.close('all')


def make_plots(evals):
    optimal_points = []
    totalEvals = []
    for totalEval_key in sorted(evals.keys()):
        batch_means = []
        batch_sizes = evals[totalEval_key]
        for batch in sorted(batch_sizes.keys()):
            batch_means.append(batch_sizes[batch]['fitness_mean'])
            # print('TOTAL: %s \t Batch: %s' %(totalEval_key, batch))
        try:
            best_value = np.argmin(batch_means)
            totalEvals.append(totalEval_key)
            optimal_points.append(batch_sizes.keys()[best_value])
            print(batch_sizes.keys())
            print(batch_means)
            plt.plot(batch_sizes.keys(), batch_means, 'ko')
            plt.title('Total evaluations: %s --- [best value = %s]' %(
                totalEval_key, batch_sizes.keys()[best_value])
            )
            plt.grid()
            plt.yscale('log')
            plt.xlabel('Chunk size')
            plt.ylabel(r"$\hat{\hat{R}}$")
            plt.savefig(
                os.path.join(RESULT_DIR, 'total_eval_%s' %totalEval_key),
                bbox_inches='tight'
            )
            plt.close('all')
        except:
            print('Missing data')
    plot_optimal_values(optimal_points, totalEvals)


def main():
    evals = collect_totalEval_scenario()
    make_plots(evals)

main()

# if __name__ == '__main__':
#     main()
