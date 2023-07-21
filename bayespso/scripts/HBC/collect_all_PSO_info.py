""" Collects the information of all the iterations together into one file for PSO """
import os
import json
import glob


BASE = '/home/user/PSO_Bayes_stability_kappa_03/PSO/*'
OUTPUT_DIR = '/home/user/HBC_analysis/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def scan_one_iteration(iteration_path):
    particles_path = os.path.join(iteration_path, '*', 'score.json')
    iteration_info = {}
    for path in glob.glob(particles_path):
        with open(path, 'rt') as score_file:
            scores = json.load(score_file)
        parameters_path = os.path.join(os.path.dirname(path), 'parameters.json')
        with open(parameters_path, 'rt') as parameters_file:
            parameters = json.load(parameters_file)
        particle_nr = int(os.path.basename(os.path.dirname(path)))
        iteration_info[particle_nr] = {
            'parameters': parameters,
            'scores': scores
        }
    return iteration_info


def one_repeat(repeat_path):
    iteration_path = os.path.join(repeat_path, 'previous_files', 'iteration_')
    repeat_info = {}
    for i in range(100):
        path = iteration_path + str(i)
        repeat_info[i] = scan_one_iteration(path)
    return repeat_info


def collect_all_repeats(algorithm):
    base_path = os.path.join(BASE, algorithm, '*')
    full_data = {}
    for path in glob.glob(BASE):
        repeat_nr = int(os.path.basename(path))
        full_data[repeat_nr] = one_repeat(path)
    results_path = os.path.join(OUTPUT_DIR, '%s_full_data.json' %algorithm)
    with open(results_path, 'wt') as out_file:
        json.dump(full_data, out_file, indent=4)


def main():
    print('Starting PSO collection')
    collect_all_repeats('PSO')
    print('Starting Bayes collection')
    collect_all_repeats('Bayes')


main()