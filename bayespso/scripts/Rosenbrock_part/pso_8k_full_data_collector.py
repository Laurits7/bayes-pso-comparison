import glob
import os
import json
import numpy as np

PSO_BASE = '/home/user/Papers/bayespso-paper-sw/data/numeric_data/Rosenbrock/'


def collect_rosenbrock_pso():
    pso_base = '/home/user/Papers/BO_PSO_DATA/Rosenbrock/PSO'
    chunk_wildcard = os.path.join(pso_base, 'chunk_*')
    total_info = {}
    for chunk_dir in glob.glob(chunk_wildcard):
        chunk_size = int(os.path.basename(chunk_dir).split('_')[-1])
        chunk_info = collect_chunkwise(chunk_dir)
        total_info[chunk_size] = chunk_info
    output_path = os.path.join(PSO_BASE, 'full_pso.json')
    with open(output_path, 'wt') as out_file:
        json.dump(total_info, out_file, indent=4)
    return total_info


def collect_chunkwise(chunk_dir):
    repeat_wildcard = os.path.join(chunk_dir, 'repeat_*')
    chunk_info = {}
    for repeat_dir in glob.glob(repeat_wildcard):
        repeat_number = int(os.path.basename(repeat_dir).split('_')[-1])
        repeat_info = collect_repeats(repeat_dir)
        chunk_info[repeat_number] = repeat_info
    return chunk_info


def collect_repeats(repeat_dir):
    iter_info_path = os.path.join(repeat_dir, 'iter_info.jsonx')
    repeat_info = {}
    with open(iter_info_path, 'rt') as in_file:
        for line in in_file:
            info = json.loads(line)
            repeat_info[info['iteration']] = info['best_fitness']
    return repeat_info

# ---------------------------------------------------------------------------

def prepare_pso_plotting(total_info):
    prepared_full_info = {}
    for chunk_size in total_info.keys():
        prepared_full_info[chunk_size] = {}
        chunk_info = total_info[chunk_size]
        for repeat_nr in chunk_info.keys():
            repeat_info = chunk_info[repeat_nr]
            for iteration_nr in repeat_info.keys():
                if not iteration_nr in list(prepared_full_info[chunk_size].keys()):
                    prepared_full_info[chunk_size][iteration_nr] = []
                iteration_info = repeat_info[iteration_nr]
                prepared_full_info[chunk_size][iteration_nr].append(iteration_info)
    output_path = os.path.join(PSO_BASE, 'collected_pso.json')
    with open(output_path, 'wt') as out_file:
        json.dump(prepared_full_info, out_file, indent=4)
    return prepared_full_info

# ---------------------------------------------------------------------------

def prepare_in_plotting_format(prepared_full_info):
    plotting_info = {}
    for chunk_size in prepared_full_info.keys():
        plotting_info[chunk_size] = []
        iterations = sorted(list(prepared_full_info[chunk_size].keys()))
        for iteration in iterations:
            bests = prepared_full_info[chunk_size][iteration]
            info = {
                'mean': np.mean(bests),
                'nr_repetitions': len(bests),
                'std': np.std(bests),
                'stderr': np.std(bests)/len(bests),
                'iteration': int(iteration)
            }
            plotting_info[chunk_size].append(info)
    output_path = os.path.join(PSO_BASE, 'plotting_ready_pso_full.json')
    with open(output_path, 'wt') as out_file:
        json.dump(plotting_info, out_file, indent=4)
    return plotting_info




def main():
    # STEP 1: collect all the date to one file:
    total_info = collect_rosenbrock_pso()
    # STEP 2: all best scores per iteration to one list
    prepared_full_info = prepare_pso_plotting(total_info)
    # STEP 3: Create the datafile used for plotting
    correct_format_data = prepare_in_plotting_format(prepared_full_info)