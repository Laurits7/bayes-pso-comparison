import os
import glob
import json
import numpy as np

BASE = '/home/user/pso_batch_test7'
OUTPUT_PATH = '/home/user/batch_study.json'


def totalEval_analysis():
    full_results = {}
    totalEval_wcp = os.path.join(BASE, "*", 'results', "*")
    for path in glob.glob(totalEval_wcp):
        totalEvals = int(os.path.basename(path))
        totalEval_info, best_batch_size = batchWise_analysis(path)
        full_results[totalEvals] = {
            'full_info': totalEval_info,
            'best_batch_size': best_batch_size,
        }
    with open(OUTPUT_PATH, 'wt') as inFile:
        json.dump(full_results, inFile, indent=4)



def batchWise_analysis(totalEvals_dir):
    totalEval_info = {}
    best_batch_value = 99e99
    best_batch_size = None
    batch_dir = os.path.join(totalEvals_dir, '*')
    for path in glob.glob(batch_dir):
        batch_size = os.path.basename(path).split('_')[-1]
        fitnesses_file = os.path.join(path, 'results', 'fitnesses.txt')
        fitnesses = []
        with open(fitnesses_file, 'rt') as inFile:
            for line in inFile:
                fitnesses.append(float(line.strip('\n')))
        if best_batch_value > np.mean(fitnesses):
            best_batch_value = np.mean(fitnesses)
            best_batch_size = batch_size
        totalEval_info[batch_size] = {
            'fitnesses': fitnesses,
            'mean': np.mean(fitnesses),
            'std': np.std(fitnesses),
            'stdev': np.std(fitnesses)/np.sqrt(len(fitnesses)),
            'nr_repetitions': len(fitnesses)
        }
    return totalEval_info, best_batch_size


totalEval_analysis()