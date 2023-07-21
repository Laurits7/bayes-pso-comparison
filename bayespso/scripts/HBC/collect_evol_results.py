import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


BAYES_BASE = '/home/user/PSO_Bayes_stability_kappa_03/Bayes'
OUTPUT_DIR = '/home/user/Stability_results'
PSO_BASE = '/home/user/PSO_Bayes_stability_kappa_03/PSO'
nr_iterations = 100

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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


def plot_performance(bayes_avg_performance, pso_avg_performance):
    bayes_x = np.arange(len(bayes_avg_performance['avg_iteration_best']))
    pso_x = np.arange(len(pso_avg_performance['avg_iteration_best']))
    ### AVG ###
    plt.plot(
        bayes_x, bayes_avg_performance['avg_iteration_best'], label='BO avg',
        c='r', ls='solid', marker='None'
    )
    plt.plot(
        pso_x, pso_avg_performance['avg_iteration_best'], label='PSO avg',
        c='g', ls='solid', marker='None'
    )
    ### TRAIN ###
    plt.plot(
        bayes_x, bayes_avg_performance['avg_corresponding_test'], label='BO test',
        c='r', ls='dotted', marker='None'
    )
    plt.plot(
        pso_x, pso_avg_performance['avg_corresponding_test'], label='PSO test',
        c='g', ls='dotted', marker='None'
    )
    ### TEST ###
    plt.plot(
        bayes_x, bayes_avg_performance['avg_corresponding_train'], label='BO train',
        c='r', ls='dashed', marker='None'
    )
    plt.plot(
        pso_x, pso_avg_performance['avg_corresponding_train'], label='PSO train',
        c='g', ls='dashed', marker='None'
    )
    ###### STYLE ######
    plt.grid()
    plt.legend(bbox_to_anchor=(1.35, 1))
    plt.ylabel('d_ams')
    plt.xlabel('Iteration')
    outpath = os.path.join(OUTPUT_DIR, 'performance_results.png')
    plt.savefig(outpath, bbox_inches='tight')
    plt.close('all')
    ### Save plotting
    output_path = os.path.join(OUTPUT_DIR, 'HBC_trainSet_evolution.json')
    to_save = {
        'Bayes': bayes_avg_performance,
        'PSO': pso_avg_performance
    }
    with open(output_path, 'wt') as out_file:
        json.dump(to_save, out_file)


def get_avg_performance(repeats):
    avg_performance = {
        'avg_iteration_best': [],
        'avg_corresponding_test': [],
        'avg_corresponding_train': []
    }
    for i in range(nr_iterations):
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
        for i in range(nr_iterations):
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


def calculate_train(d_ams, test_ams, kappa=0.3):
    return (test_ams - d_ams)/kappa + test_ams


bayes_results = collect_results(BAYES_BASE)
bayes_avg_perf = get_avg_performance(bayes_results)
pso_results = collect_results(PSO_BASE)
pso_avg_perf = get_avg_performance(pso_results)
plot_performance(bayes_avg_perf, pso_avg_perf)
