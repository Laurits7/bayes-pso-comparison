import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np


PSO_WILDCARD = '/home/user/Papers/BO_PSO_DATA/Rosenbrock/PSO/chunk_100/*/iter_info.jsonx'
BAYES_WILDCARD = '/home/user/Papers/BO_PSO_DATA/Rosenbrock/bayes8k/batch_100/*/locations.json'
MAX_EVAL = 8000
CHUNK_SIZE = 100
MAX_ITER = int(MAX_EVAL / CHUNK_SIZE)
############################################
def access_one_file(path):
    best_values = []
    best_value = 99e99
    with open(path, 'rt') as inFile:
        for i, line in enumerate(inFile):
            if i > MAX_ITER:
                break
            best_fitness = json.loads(line)['best_fitness']
            if best_fitness < best_value:
                best_value = best_fitness
            best_values.append(best_value)
    return best_values


def collect_iterwise(wild_card):
    iter_values = {}
    repeat_bests = []
    for path in glob.glob(wild_card):
        repeat_bests.append(access_one_file(path))
    for repeat_best in repeat_bests:
        for i, entry in enumerate(repeat_best):
            if i not in iter_values.keys():
                iter_values[i] = []
            iter_values[i].append(entry)
    return iter_values


def plot_pso_bayes():
    pso_means = []
    pso_maxs = []
    pso_mins = []
    bayes_means = []
    bayes_maxs = []
    bayes_mins = []
    pso_iter_values = collect_iterwise(PSO_WILDCARD)
    bayes_iter_values = collect_bayes_iterwise(BAYES_WILDCARD)
    pso_bayes_distrib(pso_iter_values, bayes_iter_values)
    for i in range(MAX_ITER):
        pso_mean = np.mean(pso_iter_values[i])
        pso_stdev = np.std(pso_iter_values[i])
        pso_mins.append(max(0, pso_mean - pso_stdev))
        pso_maxs.append(pso_mean + pso_stdev)
        pso_means.append(pso_mean)
        ###################################
        bayes_mean = np.mean(bayes_iter_values[i])
        bayes_stdev = np.std(bayes_iter_values[i])
        bayes_mins.append(max(0, bayes_mean - bayes_stdev))
        bayes_maxs.append(bayes_mean + bayes_stdev)
        bayes_means.append(bayes_mean)
    print('PSO: ' + str(pso_mean) + ' stdev: ' + str(pso_stdev))
    print('Bayes: ' + str(bayes_mean) + ' stdev: ' + str(bayes_stdev))
    number_evals = list(np.arange(1, MAX_ITER + 1) * CHUNK_SIZE)
    plt.plot(number_evals, pso_means, label='PSO_chunk100')
    plt.fill_between(number_evals, pso_mins, pso_maxs, alpha=0.2)
    plt.plot(number_evals, bayes_means, label='bayes_chunk100')
    plt.fill_between(number_evals, bayes_mins, bayes_maxs, alpha=0.2)
    output_path = '/home/user/tmp3/pso_bayes_8k_mean.png'
    plt.legend()
    plt.grid()
    plt.xlabel('Evaluations')
    plt.ylabel(r"$\hat{\hat{R}}$")
    plt.xlim(0, MAX_EVAL)
    plt.yscale('log')
    plt.title('PSO evolution mean')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')


def pso_bayes_distrib(pso_iter_values, bayes_iter_values):
    out_path = '/home/user/tmp3/pso_bayes_8k_distrib.png'
    number_evals = list(np.arange(MAX_ITER) * CHUNK_SIZE)
    print(len(pso_iter_values))
    pso_values = pso_iter_values[MAX_ITER - 1]
    bayes_values = bayes_iter_values[MAX_ITER - 1]
    bins = np.logspace(np.log10(1e-4), np.log10(1e4), 32)
    plt.hist(pso_values, bins=bins, alpha=0.3, label='PSO_chunk100')
    plt.hist(bayes_values, bins=bins, alpha=0.3, label='Bayes_chunk100')
    plt.legend()
    plt.xlim((1e-4, 1e4))
    plt.xscale('log')
    plt.xlabel(r"$\hat{\hat{R}}$")
    plt.ylabel("Number of trials per bin")
    plt.title(
        'PSO and Bayes best found location')
    plt.grid()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close('all')


def access_bayes_file(path):
    function_values = []
    iterations = []
    minimums = []
    minimum = 99e99
    with open(path, 'rt') as in_file:
        for i, line in enumerate(in_file):
            if i > MAX_ITER:
                break
            d = json.loads(line)
            iterations.append(d['iteration'])
            function_values = ([e[0] for e in d['function_values']])
            if min(function_values) < minimum:
                minimum = min(function_values)
            minimums.append(minimum)
    return minimums


def collect_bayes_iterwise(wild_card):
    iter_values = {}
    repeat_bests = []
    for path in glob.glob(wild_card):
        repeat_bests.append(access_bayes_file(path))
    for repeat_best in repeat_bests:
        for i, entry in enumerate(repeat_best):
            if i not in iter_values.keys():
                iter_values[i] = []
            iter_values[i].append(entry)
    return iter_values


plot_pso_bayes()
