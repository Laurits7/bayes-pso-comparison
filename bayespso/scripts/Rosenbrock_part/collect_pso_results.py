""" This script is used to collect all the repetition info from PSO
optimization into one file for easier plotting
Call with 'python'

Usage: collect_pso_results.py.py --input_dir=DIR --output_dir=DIR

Options:
    -i --input_dir=DIR              Directory of the input
    -o --output_dir=DIR             Directory of the output
"""
import os
import glob
import json
import docopt
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

CHUNK_SIZES = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
TOTAL_REPEATS = 1000
TOTAL_EVALUATIONS = 10000


def main(input_dir, output_dir):
    base_dir = os.path.join(input_dir, 'PSO')
    collected_repeats_dir = os.path.join(output_dir, 'plots')
    output_path = os.path.join(collected_repeats_dir, 'collected_plots')
    if not os.path.exists(collected_repeats_dir):
        os.makedirs(collected_repeats_dir)
    iter_dicts = []
    plt.figure(figsize=(16, 9))
    for chunk_size in CHUNK_SIZES:
        chunk_name = 'chunk_%s' %chunk_size
        nr_iter = int(TOTAL_EVALUATIONS / chunk_size)
        iter_dict = {iteration: [] for iteration in range(nr_iter+2)}
        iter_dict = chunk_wize_collection(base_dir, chunk_name, iter_dict)
        iter_dicts.append(iter_dict)
        save_iter_dict(iter_dict, chunk_name, output_dir)
        create_plots(iter_dict, chunk_name, nr_iter)
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.title('PSO evolution with different chunk sizes')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')
    create_distrib_plots(iter_dicts, collected_repeats_dir)
    create_eval_wise_plot(iter_dicts, collected_repeats_dir)
    save_distrib_info(iter_dicts, collected_repeats_dir)


def save_distrib_info(iter_dicts, collected_repeats_dir):
    output_path = os.path.join(collected_repeats_dir, 'best_value_info.json')
    info = {}
    for chunk_size, iter_dict in zip(CHUNK_SIZES, iter_dicts):
        best_values = iter_dict[max(iter_dict.keys())]
        info[chunk_size] = {
            'mean': np.mean(best_values), 'stdev': np.std(best_values)
        }
    with open(output_path, 'wt') as out_file:
        json.dump(info, out_file, indent=4)


def create_distrib_plots(iter_dicts, collected_repeats_dir):
    output_path = os.path.join(collected_repeats_dir, 'best_distributions.png')
    bins = np.logspace(np.log10(1e-13), np.log10(1e13), 32)
    plt.figure(figsize=(16, 9))
    for chunk_size, iter_dict in zip(CHUNK_SIZES, iter_dicts):
        chunk_name = 'chunk_%s' %chunk_size
        best_values = iter_dict[max(iter_dict.keys())]
        plt.hist(best_values, bins=bins, alpha=0.3, label=chunk_name)
    plt.legend()
    plt.xlim((1e-13, 1e13))
    plt.xscale('log')
    plt.xlabel(r"$\hat{\hat{R}}$")
    plt.ylabel("Number of trials per bin")
    plt.title(
        'PSO best found location distributions with different chunk sizes')
    plt.grid()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')


def create_eval_wise_plot(iter_dicts, collected_repeats_dir):
    plt.figure(figsize=(16,9))
    output_path = os.path.join(collected_repeats_dir, 'iter')
    for chunk_size, iter_dict in zip(CHUNK_SIZES, iter_dicts):
        chunk_name = 'chunk_%s' %chunk_size
        iterations = list(iter_dict.keys())
        number_evals = list(np.array(list(iter_dict.keys())) * chunk_size)
        means = []
        mins = []
        maxs = []
        for i in iterations:
            iter_values = iter_dict[i]
            means.append(np.mean(iter_dict[i]))
            mins.append(max(0, np.mean(iter_dict[i]) - np.std(iter_dict[i])))
            maxs.append(np.mean(iter_dict[i]) + np.std(iter_dict[i]))
        plt.plot(number_evals, means, label=chunk_name)
        plt.fill_between(number_evals, mins, maxs, alpha=0.2)
    plt.legend()
    plt.grid()
    plt.xlabel('Evaluations')
    plt.ylabel(r"$\hat{\hat{R}}$")
    plt.xlim(0, 10000)
    plt.yscale('log')
    plt.title('PSO evolution with different chunk sizes')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')


def chunk_wize_collection(base_dir, chunk_name, iter_dict):
    for repeat in range(TOTAL_REPEATS):
        repeat_str = 'repeat_%s' %repeat
        wild_card_path = os.path.join(
            base_dir, chunk_name, repeat_str, 'iter_info.jsonx')
        for path in glob.glob(wild_card_path):
            best_value = 99e99
            with open(path, 'rt') as info_file:
                for line in info_file:
                    d = json.loads(line)
                    best_fitness = d['best_fitness']
                    if best_fitness < best_value:
                        best_value = best_fitness
                    iter_dict[d['iteration']].append(best_value)
    return iter_dict


def save_iter_dict(iter_dict, chunk_name, output_dir):
    collected_repeats_dir = os.path.join(output_dir, 'collected_repeats')
    if not os.path.exists(collected_repeats_dir):
        os.makedirs(collected_repeats_dir)
    output_path = os.path.join(collected_repeats_dir, '%s.json' %chunk_name)
    with open(output_path, 'wt') as collection_file:
        json.dump(iter_dict, collection_file, indent=4)


def create_plots(iter_dict, chunk_name, nr_iter):
    means = []
    stdevs = []
    mins = []
    maxs = []
    for i in range(nr_iter):
        means.append(np.mean(iter_dict[i]))
        stdevs.append(np.std(iter_dict[i]))
        mins.append(max(0, np.mean(iter_dict[i]) - np.std(iter_dict[i])))
        maxs.append(np.mean(iter_dict[i]) + np.std(iter_dict[i]))
    iterations = np.arange(nr_iter)
    plt.plot(iterations, means, label=chunk_name)
    plt.fill_between(iterations, mins, maxs, alpha=0.2)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        input_dir = arguments['--input_dir']
        main(input_dir, output_dir)
    except docopt.DocoptExit as e:
        print(e)
