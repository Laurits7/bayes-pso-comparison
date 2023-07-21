"""Call with 'python'

Usage: bayes_vs_pso.py --input_dir=DIR --output_dir=DIR

Options:
    -i --input_dir=DIR              Directory of the input
    -o --output_dir=DIR             Directory of the output
"""

import csv
import os
import docopt
import glob
import pandas
import json
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt


PSO_CHUNK = 100
BAYES_CHUNK = 1

def main(input_dir, output_dir):
    out_dir = os.path.join(output_dir, 'results')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    bayes_results, bayes_mins = collect_bayes(input_dir)
    pso_results, pso_mins = collect_pso(input_dir)
    plot_evolution(pso_results, bayes_results, out_dir)
    plot_distribution(bayes_mins, pso_mins, out_dir)



def collect_bayes(input_dir):
    all_minimums = []
    wild_card_path = os.path.join(
        input_dir, 'BO_2k_different_chunks', 'batch_%s' %BAYES_CHUNK, '*', 'locations.json')
    for path in glob.glob(wild_card_path):
        all_minimums.append(analyze_location_file(path))
    intermediate = {}
    iterations = []
    final_mins = []
    for repeat in all_minimums:
        final_mins.append(repeat[-1])
        for i, minimum in enumerate(repeat):
            if not i in intermediate:
                intermediate[i] = []
            intermediate[i].append(minimum)
    result = {'min': [], 'max': [], 'mean': []}
    for iteration in range(len(intermediate)):
        std = np.std(intermediate[iteration])
        mean = np.mean(intermediate[iteration])
        result['min'].append(mean - std)
        result['max'].append(mean + std)
        result['mean'].append(mean)
    return result, final_mins


def analyze_location_file(path):
    function_values = []
    minimums = []
    minimum = 99e99
    with open(path, 'rt') as in_file:
        for line in in_file:
            d = json.loads(line)
            function_values = ([e[0] for e in d['function_values']])
            if min(function_values) < minimum:
                minimum = min(function_values)
            minimums.append(minimum)
    return minimums


def collect_pso(input_dir):
    all_minimums = []
    wild_card_path = os.path.join(
        input_dir, 'PSO', 'chunk_%s' %PSO_CHUNK, '*', 'iter_info.jsonx')
    for path in glob.glob(wild_card_path):
        all_minimums.append(analyze_iter_info(path))
    intermediate = {}
    iterations = []
    final_mins = []
    for repeat in all_minimums:
        final_mins.append(repeat[-1])
        for i, minimum in enumerate(repeat):
            if not i in intermediate:
                intermediate[i] = []
            intermediate[i].append(minimum)
    result = {'min': [], 'max': [], 'mean': []}
    for iteration in range(len(intermediate)):
        std = np.std(intermediate[iteration])
        mean = np.mean(intermediate[iteration])
        result['min'].append(mean - std)
        result['max'].append(mean + std)
        result['mean'].append(mean)
    return result, final_mins


def analyze_iter_info(path):
    values = []
    best_value = 99e99
    with open(path, 'rt') as in_file:
        for line in in_file:
            best_fitness = json.loads(line)['best_fitness']
            if best_fitness < best_value:
                best_value = best_fitness
            values.append(best_value)
    return values


def plot_evolution(pso_results, bayes_results, out_dir):
    plot_path = os.path.join(out_dir, 'pso_vs_bayes_evol.png')
    plt.figure(figsize=(16,9))

    iterations = list(np.arange(len(pso_results['min']) + 1) * PSO_CHUNK)
    pso_results['max'].append(pso_results['max'][-1])
    pso_results['min'].append(pso_results['min'][-1])
    pso_results['mean'].append(pso_results['mean'][-1])
    plt.plot(iterations, pso_results['mean'], label='PSO_chunk_%s' %(PSO_CHUNK))
    plt.fill_between(iterations, pso_results['min'], pso_results['max'], alpha=0.2)



    iterations = list(np.arange(len(bayes_results['min']) + 1) * BAYES_CHUNK)
    bayes_results['max'].append(bayes_results['max'][-1])
    bayes_results['min'].append(bayes_results['min'][-1])
    bayes_results['mean'].append(bayes_results['mean'][-1])
    plt.plot(iterations, bayes_results['mean'], label='Bayes_chunk_%s' %BAYES_CHUNK)
    plt.fill_between(iterations, bayes_results['min'], bayes_results['max'], alpha=0.2)


    plt.grid()
    plt.xlabel('Evaluations')
    plt.ylabel(r"$\hat{\hat{R}}$")
    plt.legend()
    plt.yscale('log')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close('all')


def plot_distribution(bayes_mins, pso_mins, out_dir):
    plot_path = os.path.join(out_dir, 'pso_vs_bayes_distrib.png')
    bins = np.logspace(np.log10(1e-13), np.log10(1e13), 32)
    plt.hist(bayes_mins, bins=bins, label='Bayes')
    plt.hist(pso_mins, bins=bins, label='PSO')
    plt.legend()
    plt.xlim((1e-13, 1e13))
    plt.xscale('log')
    plt.xlabel(r"$\hat{\hat{R}}$")
    plt.ylabel("Number of trials per bin")
    plt.title(
        'Best found location distributions with Bayes and PSO')
    plt.grid()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close('all')

if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        input_dir = arguments['--input_dir']
        main(input_dir, output_dir)
    except docopt.DocoptExit as e:
        print(e)