import glob
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
import ast


BASE = '/home/user/PSO_batch_size_vol2'
RESULT_DIR = '/home/user/Paper_stuff/median_pso_batch'

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)




def collect_batches(total_eval_dir):
    batch_sizes = []
    medians = []
    batch_wcp = os.path.join(total_eval_dir, 'PSO', '*')
    for path in glob.glob(batch_wcp):
        batch_sizes.append(int(os.path.basename(path).split('_')[-1]))
        bests = load_files(path)
        medians.append(np.median(bests))
    info = {
        'batch_sizes': batch_sizes,
        'medians': medians
    }
    return info


def load_files(batch_path):
    repeat_wcp = os.path.join(batch_path, '*', 'iter_info.jsonx')
    bests = []
    for path in glob.glob(repeat_wcp):
        history = read_hist_file(path)
        best = 99e99
        for entry in history:
            best_fitness = entry['best_fitness']
            if best_fitness < best:
                best = best_fitness
        bests.append(best)
    return bests


def collect_total_evals():
    total_eval_wcp = os.path.join(BASE, '*')
    results = {}
    for path in glob.glob(total_eval_wcp):
        total_eval_info = collect_batches(path)
        total_evals = int(os.path.basename(path).split('_')[-1])
        results[total_evals] = total_eval_info
    output_path = os.path.join(RESULT_DIR, 'all_results.json')
    with open(output_path, 'wt') as out_file:
        json.dump(results, out_file)
    return results


def read_hist_file(path):
    i = 0
    objects = []
    obj = ''
    with open(path, 'rt') as inFile:
        for line in inFile:
            if i % 4 == 0 and i != 0:
                objects.append(ast.literal_eval(obj.replace('\n', '')))
                obj = '{'
            else:
                obj += line.replace(' ', '')
            i += 1
    return objects


def print_info(totalEvals, sqrt_results, one_percent_results, optimal_points):
    print('1% hypo \t | sqrt hypo \t \t| Total evaluations \t | Optimal')
    print('-----------------------------------------------------------')
    for i, j, k, l in zip(one_percent_results, sqrt_results, totalEvals, optimal_points):
        print('%s \t \t | %s \t \t | %s \t | %s' %(i, round(j, 2), k, l))


def plot_optimal_values(optimal_points, totalEvals):
    sqrt_results = [np.sqrt(evalSize) for evalSize in sorted(totalEvals)]
    one_percent_results = [evalSize/100 for evalSize in sorted(totalEvals)]
    print_info(totalEvals, sqrt_results, one_percent_results, optimal_points)
    plt.plot(
        totalEvals, optimal_points, color='k', marker='',
        label='True optimal points'
    )
    plt.plot(sorted(totalEvals), sqrt_results, color='k', marker='^',
        label='Sqrt hypothesis', ls=''
    )
    plt.plot(sorted(totalEvals), one_percent_results, color='k', marker='s',
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


def plotting(results):
    optimal_points = []
    total_evals = sorted(list(results.keys()))
    for result in total_evals:
        batch_info = results[result]
        best_value_idx = np.argmin(batch_info['medians'])
        plt.plot(batch_info['batch_sizes'], batch_info['medians'], 'ko')
        plt.title('Total evaluations: %s --- [best value = %s]' %(
            result, batch_info['batch_sizes'][best_value_idx])
        )
        plt.grid()
        plt.yscale('log')
        plt.xlabel('Chunk size')
        plt.ylabel(r"$\hat{\hat{R}}$")
        plt.savefig(
            os.path.join(RESULT_DIR, 'total_eval_%s' %result),
            bbox_inches='tight'
        )
        plt.close('all')
        optimal_points.append(batch_info['batch_sizes'][best_value_idx])
    plot_optimal_values(optimal_points, total_evals)

def main():
    results = collect_total_evals()
    plotting(results)


main()