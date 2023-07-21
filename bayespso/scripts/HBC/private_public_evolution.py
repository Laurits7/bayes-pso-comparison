""" Collects the information of all the iterations together into one file for PSO """
import os
import json
import glob


BASE = '/home/user/PSO_Bayes_stability_kappa_03/'
OUTPUT_DIR = '/home/user/HBC_analysis/collected_evol_Kaggle'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def one_repeat(repeat_path):
    iteration_path = os.path.join(repeat_path, 'previous_files', 'iteration_')
    repeat_best_score = -99e99
    evolution = []
    for i in range(100):
        path = iteration_path + str(i)
        best_score, best_path = scan_one_iteration(path)
        if best_score > repeat_best_score:
            repeat_best_score = best_score
            progress = {
                'iteration': i,
                'path': best_path,
                'score': best_score
            }
            evolution.append(progress)
    return evolution


def scan_one_iteration(iteration_path):
    particles_path = os.path.join(iteration_path, '*', 'score.json')
    best_path = None
    best_score = -99e99
    for path in glob.glob(particles_path):
        with open(path, 'rt') as score_file:
            d_ams = (-1) * json.load(score_file)['d_ams']
        if d_ams > best_score:
            best_score = d_ams
            best_path = os.path.join(os.path.dirname(path), 'parameters.json')
    return best_score, best_path


def collect_all_repeats(algorithm):
    evolutions = {}
    search_dir = os.path.join(BASE, algorithm, '*')
    for path in glob.glob(search_dir):
        repeat = os.path.basename(path)
        evolutions[repeat] = one_repeat(path)
    results_path = os.path.join(OUTPUT_DIR, '%s_results.json' %algorithm)
    with open(results_path, 'wt') as out_file:
        json.dump(evolutions, out_file)
    return evolutions


def main():
    pso_evolutions = collect_all_repeats('PSO')
    bayes_evolutions = collect_all_repeats('Bayes')


if __name__ == '__main__':
    main()