''' Plots the seed impact'''

import matplotlib.pyplot as plt
import glob
import json
import os


BAYES_PATH = '/home/user/Bayes_PSO/HBC_stability/bo_results.json'
PSO_PATH = '/home/user/Bayes_PSO/HBC_stability/pso_results.json'
PSO_PATH2 = '/home/user/Bayes_PSO/HBC_stability/pso_results.json_'



def main():
    with open(BAYES_PATH, 'rt') as in_file:
        bayes_results = json.load(in_file)
    with open(PSO_PATH, 'rt') as in_file:
        pso_results = json.load(in_file)
    plt.figure(figsize=(16,9))
    plt.hist(pso_results['public'], color='b', alpha=0.3, hatch='\\\\', label='PSO_public')
    plt.hist(pso_results['private'], color='b', alpha=0.3, hatch='//', label='PSO_private')
    plt.hist(bayes_results['public'], color='r', alpha=0.3, hatch='||', label='Bayes_public')
    plt.hist(bayes_results['private'], color='r', alpha=0.3, hatch='--', label='Bayes_private')
    plt.title('PSO and BO stability')
    plt.grid()
    plt.legend()
    plt.savefig('/home/user/tmp4/pso_bayes_HBC_stability', bbox_inches='tight')
    plt.close('all')



if __name__ == '__main__':
    main()