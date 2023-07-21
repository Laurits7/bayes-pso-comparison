import os
import glob
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib

RESULTS_WCP = '/home/user/Desktop/BayesPSO/Papers/BayesPSO/bayespso-paper-sw/data/Final_run/PSO_batch_study/3000/batch_*'


def collect_all_results():
    total_results = {}
    for batch_dir in glob.glob(RESULTS_WCP):
        batch_size = int(batch_dir.split('/')[-1].split('_')[1])
        total_results[batch_size] = collect_batch_results(batch_dir)
    return total_results


def collect_batch_results(batch_dir):
    repeats_wcp = os.path.join(batch_dir, 'results', 'repeat_*', 'evol.json')
    iter_values = {}
    for repeat_path in glob.glob(repeats_wcp):
        with open(repeat_path, 'rt') as inFile:
            for i, line in enumerate(inFile):
                if i+1 not in iter_values.keys():
                    iter_values[i+1] = []
                iter_values[i+1].append(json.loads(line)['best_fitness'])
    batch_results = {}
    for iteration, all_values in iter_values.items():
        batch_results[iteration] = {
            'stdev': np.std(all_values),
            'mean': np.mean(all_values),
            'stderr': np.std(all_values)/np.sqrt(len(all_values)),
            'n_repeats': len(all_values)
        }
    return batch_results

all_results = collect_all_results()

xs = []
yerr = []
ys = []
ordered_batch_sizes = sorted(all_results.keys())
for batch_size in ordered_batch_sizes:
    info = all_results[batch_size]
    last_iteration = max(info.keys())
    batch_best = info[last_iteration]
    ys.append(batch_best['mean'])
    xs.append(batch_size)
    yerr.append(batch_best['stderr'])

fig, ax = plt.subplots()
ax.fill_between(
    xs,
    np.array(np.array(ys) - np.array(yerr)),
    np.array(np.array(ys) + np.array(yerr)),
    color='b',
    alpha=0.3
)
ax.plot(xs, ys, marker='o', mfc='none', mec='k', ms=5)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel(r"$\hat{\hat{R}}$")
ax.set_xlabel(r'$N_{parallel}$')

ax.axvline(
    3000*0.021,
    ymin=0,
    ymax=np.max(np.array(ys) + np.array(yerr)),
    color='r',
    ls='--',
    label=r'2%'
)
locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
ax.yaxis.set_major_locator(locmaj)
locmin = matplotlib.ticker.LogLocator(
        base=10,
        subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        numticks=12
)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

ax.grid()
plt.legend()
plt.savefig('/home/user/tmp/PSO_batch_study/total_3000.png', bbox_inches='tight')
plt.close('all')