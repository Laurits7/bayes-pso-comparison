import glob
import csv
import os
import json
import numpy as np

OUTPUT_DIR = '/home/user/Papers/BO_PSO_DATA'

def only_bayes_2k_results():
    bayes_dir = os.path.join(OUTPUT_DIR, 'Rosenbrock', 'BO_2k_different_chunks')
    bayes_chunk_sizes = []
    to_save = {}
    for path in glob.glob(os.path.join(bayes_dir, '*')):
        bayes_chunk_sizes.append(os.path.basename(path).split('_')[1])
    for chunk_size in bayes_chunk_sizes:
        result, final_mins = collect_bayes(bayes_dir, chunk_size)
        to_save[chunk_size] = {
            'evol': result,
            'final_mins': final_mins
        }
    data_file = os.path.join(OUTPUT_DIR, 'rosenbrock_bo_2k.json')
    with open(data_file, 'wt') as out_file:
        json.dump(to_save, out_file)



def collect_bayes(bayes_dir, chunk_size):
    all_minimums = []
    wild_card_path = os.path.join(
        bayes_dir, 'batch_%s' %chunk_size, '*', 'locations.json')
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

# only_bayes_2k_results()

# ----------------------------------------------------------------------------

def only_pso_8k_results():
    pso_dir = os.path.join(OUTPUT_DIR, 'Rosenbrock', 'PSO')
    pso_chunk_sizes = []
    to_save = {}
    for path in glob.glob(os.path.join(pso_dir, '*')):
        pso_chunk_sizes.append(os.path.basename(path).split('_')[1])
    for chunk_size in pso_chunk_sizes:
        result, final_mins = collect_pso(pso_dir, chunk_size)
        to_save[chunk_size] = {
            'evol': result,
            'final_mins': final_mins
        }

    data_file = os.path.join('/home/user/tmp8', 'rosenbrock_pso_8k.json')
    with open(data_file, 'wt') as out_file:
        json.dump(to_save, out_file, indent=4)


def collect_pso(input_dir, chunk):
    all_minimums = []
    wild_card_path = os.path.join(
        input_dir, 'chunk_%s' %chunk, '*', 'iter_info.jsonx')
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

only_pso_8k_results()


# ----------------------------------------------------------------------------


def only_bo_timing_results():
    bo_dir = os.path.join(OUTPUT_DIR, 'Rosenbrock', 'BO_2k_different_chunks')
    bo_chunk_sizes = []
    for path in glob.glob(os.path.join(bo_dir, '*')):
        bo_chunk_sizes.append(os.path.basename(path).split('_')[1])
    results = analyze_timing(bo_dir, bo_chunk_sizes)
    data_file = os.path.join(OUTPUT_DIR, 'rosenbrock_bo_timing.json')
    with open(data_file, 'wt') as out_file:
        json.dump(results, out_file)


def analyze_timing(input_dir, chunk_sizes):
    results = {}
    for chunk_size in chunk_sizes:
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
        results[chunk_size] = result
    return results


def read_timing(path):
    times = []
    with open(path, 'rt') as in_file:
        for line in in_file:
            time = json.loads(line)['time']
            times.append(time)
    return times

# only_bo_timing_results()

# ----------------------------------------------------------------------------

def bo_vs_pso():
    pso_result, pso_final_mins = collect_pso()
    pso_result = {
        'evol': pso_result,
        'final_mins': pso_final_mins
    }
    bayes_result = only_bayes_8k_results()
    to_save = {
        'pso': pso_result,
        'bo': bayes_result
    }
    data_file = os.path.join(OUTPUT_DIR, 'pso_vs_bo.json')
    with open(data_file, 'wt') as out_file:
        json.dump(to_save, out_file)

def collect_pso():
    all_minimums = []
    wild_card_path = os.path.join(
        OUTPUT_DIR, 'Rosenbrock', 'PSO', 'chunk_%s' %100, '*', 'iter_info.jsonx')
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


def only_bayes_8k_results():
    bayes_dir = os.path.join(OUTPUT_DIR, 'Rosenbrock', 'bayes8k')
    result, final_mins = collect_bayes(bayes_dir, 100)
    to_save = {
        'evol': result,
        'final_mins': final_mins
    }
    return to_save

# bo_vs_pso()

# ----------------------------------------------------------------------------
# NOT FINISHED YET


BAYES_BASE = '/home/user/PSO_Bayes_stability/Bayes*' # needs altering
OUTPUT_DIR = '/home/user/' # needs altering
PSO_BASE = '/home/user/PSO_Bayes_stability/PSO_N' # needs altering


def get_avg_performance(repeats):
    avg_performance = {
        'avg_iteration_best': [],
        'avg_corresponding_test': [],
        'avg_corresponding_train': []
    }
    for i in range(100):
        iteration_best_values = []
        corresponding_train_values = []
        corresponding_test_values = []
        for key in repeats:
            iteration_best_values.append(repeats[key]['iteration_bests'][i])
            corresponding_train_values.append(repeats[key]['corresponding_train'][i])
            corresponding_test_values.append(repeats[key]['corresponding_test'][i])
        avg_performance['avg_iteration_best'].append(np.mean(iteration_best_values))
        avg_performance['avg_corresponding_test'].append(np.mean(corresponding_test_values))
        avg_performance['avg_corresponding_train'].append(np.mean(corresponding_train_values))
    return avg_performance


def collect_results(base):
    finished_runs = os.path.join(base, '*', 'best_hyperparameters.json')
    repeats = {}
    for path in glob.glob(finished_runs):
        run_dir = os.path.dirname(path)
        repeat_nr = run_dir.split('/')[-1]
        iteration_bests = []
        corresponding_test_amss = []
        corresponding_train_amss = []
        global_best = 0
        for i in range(100):
            wcp = os.path.join(run_dir, 'previous_files', 'iteration_%s' %i, '*', 'score.json')
            max_ = 0
            for path in glob.glob(wcp):
                with open(path, 'rt') as in_file:
                    score_d = json.load(in_file)
                    score = score_d['d_ams']
                    score *= -1
                    if score > max_:
                        max_ = score
                        test_ams = (-1)*score_d['test_ams']
                        train_ams = calculate_train(score, test_ams)
            if max_ > global_best:
                global_best = max_
                corresponding_test_ams = test_ams
                corresponding_train_ams = train_ams
            print('%s ::::::::: %s' %(i, global_best))
            iteration_bests.append(global_best)
            corresponding_test_amss.append(corresponding_test_ams)
            corresponding_train_amss.append(corresponding_train_ams)
        repeats[repeat_nr] = {
            'iteration_bests': iteration_bests,
            'corresponding_test': corresponding_test_amss,
            'corresponding_train': corresponding_train_amss
        }
    return repeats



def collect_bayes_results():
    finished_runs = os.path.join(BAYES_BASE, '*', 'best_hyperparameters.json')
    repeats = {}
    for path in glob.glob(finished_runs):
        run_dir = os.path.dirname(path)
        repeat_nr = run_dir.split('/')[-1]
        location_file = os.path.join(run_dir, 'locations.json')
        with open(location_file, 'rt') as in_file:
            global_best = 0
            evolution = []
            for i, line in enumerate(in_file):
                function_values = json.loads(line)['function_values']
                function_values = [(-1)*i[0] for i in function_values]
                iteration_max = max(function_values)
                if iteration_max > global_best:
                    global_best = iteration_max
                evolution.append(global_best)
        repeats[repeat_nr] = evolution
    return repeats


def save_HBCtrain_results():
    bayes_results = collect_results(BAYES_BASE)
    bayes_avg_perf = get_avg_performance(bayes_results)
    pso_results = collect_results(PSO_BASE)
    pso_avg_perf = get_avg_performance(pso_results)
    to_save = {
        'pso': pso_avg_perf,
        'bo': bayes_avg_perf
    }
    data_file = os.path.join(OUTPUT_DIR, 'hbc_pso_bo_train.json')
    with open(data_file, 'wt') as out_file:
        json.dump(to_save, out_file)

save_HBCtrain_results()