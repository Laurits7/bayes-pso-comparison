"""Call with 'python'

Usage: plot_bayes.py --input_dir=DIR --output_dir=DIR

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

CHUNK_SIZES = [1, 5, 25, 50, 100, 250, 500, 1000]


def main(input_dir, output_dir):
    out_dir = os.path.join(output_dir, 'results')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    analyze_locations(input_dir, output_dir)
    analyze_timing(input_dir, out_dir)


def analyze_timing(input_dir, output_dir):
    results = []
    for chunk_size in CHUNK_SIZES:
        timings = []
        wild_card_path = os.path.join(
            input_dir, 'batch_%s' %chunk_size, '*', 'time.json')
        for path in glob.glob(wild_card_path):
            timings.append(read_timing(path))
        intermediate = {}
        for timing in timings:
            for i, time in enumerate(timing):
                if not i in intermediate:
                    intermediate[i] = []
                intermediate[i].append(time)
        result = {'min': [], 'max': [], 'mean': [], 'cumulative': []}
        for iteration in range(len(intermediate)):
            std = np.std(intermediate[iteration])
            mean = np.mean(intermediate[iteration])
            result['min'].append(mean - std)
            result['max'].append(mean + std)
            result['mean'].append(mean)
            if len(result['cumulative']) == 0:
                result['cumulative'].append(mean)
            else:
                result['cumulative'].append(float(result['cumulative'][-1]) + float(mean))
        results.append(result)
    plot_timings(results, output_dir)


def plot_timings(results, output_dir):
    fig_out = os.path.join(output_dir, 'timings.png')
    plt.figure(figsize=(16, 9))
    for result, chunk_size in zip(results, CHUNK_SIZES):
        iterations = list(np.arange(len(result['min']) + 1) * chunk_size)
        result['max'].append(result['max'][-1])
        result['min'].append(result['min'][-1])
        result['mean'].append(result['mean'][-1])
        result['cumulative'].append(result['cumulative'][-1])
        chunk_name = 'chunk_%s' %chunk_size
        plt.plot(iterations, result['mean'], label=chunk_name)
        plt.fill_between(iterations, result['min'], result['max'], alpha=0.2)
    plt.grid()
    plt.xlabel('Evaluations')
    plt.ylabel("time [s]")
    plt.legend()
    plt.yscale('log')
    plt.xlim(0, 2000)
    plt.savefig(fig_out, bbox_inches='tight')
    plt.close('all')

    fig_out = os.path.join(output_dir, 'cumulative_timings.png')
    plt.figure(figsize=(16, 9))
    for result, chunk_size in zip(results, CHUNK_SIZES):
        iterations = list(np.arange(len(result['min'])) * chunk_size)
        chunk_name = 'chunk_%s' %chunk_size
        plt.plot(iterations, result['cumulative'], label=chunk_name)
    plt.grid()
    plt.xlabel('Evaluations')
    plt.ylabel("time [s]")
    plt.legend()
    plt.yscale('log')
    plt.xlim(0, 2000)
    plt.savefig(fig_out, bbox_inches='tight')
    plt.close('all')


def read_timing(path):
    times = []
    with open(path, 'rt') as in_file:
        for line in in_file:
            time = json.loads(line)['time']
            times.append(time)
    return times


def analyze_locations(input_dir, output_dir):
    results = []
    chunk_finals = []
    for chunk_size in CHUNK_SIZES:
        all_minimums = []
        wild_card_path = os.path.join(
            input_dir, 'batch_%s' %chunk_size, '*', 'locations.json')
        for path in glob.glob(wild_card_path):
            all_minimums.append(analyze_location_file(path, chunk_size))
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
        results.append(result)
        chunk_finals.append(final_mins)
    plot_locations(results, output_dir)
    plot_distributions(chunk_finals, output_dir)


def plot_locations(results, output_dir):
    fig_out = os.path.join(output_dir, 'evolution.png')
    plt.figure(figsize=(16, 9))
    for result, chunk_size in zip(results, CHUNK_SIZES):
        iterations = list(np.arange(len(result['min']) + 1) * chunk_size)
        chunk_name = 'chunk_%s' %chunk_size
        result['max'].append(result['max'][-1])
        result['min'].append(result['min'][-1])
        result['mean'].append(result['mean'][-1])
        plt.plot(iterations, result['mean'], label=chunk_name)
        plt.fill_between(iterations, result['min'], result['max'], alpha=0.2)
    plt.grid()
    plt.xlabel('Evaluations')
    plt.ylabel(r"$\hat{\hat{R}}$")
    plt.legend()
    plt.yscale('log')
    plt.xlim(0, 2000)
    plt.savefig(fig_out, bbox_inches='tight')
    plt.close('all')


def plot_distributions(chunk_finals, output_dir):
    fig_out = os.path.join(output_dir, 'result_distribution.png')
    plt.figure(figsize=(16, 9))
    bins = np.logspace(np.log10(1e-2), np.log10(1e6), 32)
    for final, chunk_size in zip(chunk_finals, CHUNK_SIZES):
        chunk_name = 'chunk_%s' %chunk_size
        plt.hist(final, bins=bins, alpha=0.3, label=chunk_name)
    plt.legend()
    plt.xlim((1e-2, 1e6))
    plt.xscale('log')
    plt.xlabel(r"$\hat{\hat{R}}$")
    plt.ylabel("Number of trials per bin")
    plt.title(
        'Bayes optimization best found location distributions with different batch sizes')
    plt.grid()
    plt.savefig(fig_out, bbox_inches='tight')
    plt.close('all')


def analyze_location_file(path, chunk_size):
    function_values = []
    iterations = []
    minimums = []
    minimum = 99e99
    with open(path, 'rt') as in_file:
        for line in in_file:
            d = json.loads(line)
            iterations.append(d['iteration'])
            function_values = ([e[0] for e in d['function_values']])
            if min(function_values) < minimum:
                minimum = min(function_values)
            minimums.append(minimum)
    return minimums



if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        input_dir = arguments['--input_dir']
        main(input_dir, output_dir)
    except docopt.DocoptExit as e:
        print(e)