"""Call with 'python'

Usage: analyze_multirun_stability_results.py --output_dir=DIR

Options:
    -o --output_dir=DIR             Directory of the output
"""
import docopt
import glob
import os
import json
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from bayespso.tools import kaggle_score_calculator as ksc
from bayespso.tools import submission_higgs as sh

PATH_TO_TRUTH = '/home/user/atlas-higgs-challenge-2014-v2.csv'
BAYES_BASE = '/home/user/PSO_Bayes_stability_kappa_03/Bayes/*/'
PSO_BASE = '/home/user/PSO_Bayes_stability_kappa_03/PSO/*/'
PATH_TO_TRAIN = '/home/user/training.csv'
PATH_TO_TEST = '/home/user/test.csv'
HYPERPARAMETER_RANGES = {
    "num_boost_round": {"min": 1, "max": 500},
    "learning_rate": {"min": 1e-5, "max": 1},
    "max_depth": {"min": 1, "max": 6},
    "gamma": {"min": 0, "max": 5},
    "min_child_weight": {"min": 0, "max": 500},
    "subsample": {"min": 0.8, "max": 1},
    "colsample_bytree": {"min": 0.3, "max": 1}
}


def normalize_hyperparameters(hyperparameter_set):
    normalized_hyperparameters = {}
    for key in hyperparameter_set.keys():
        val = hyperparameter_set[key]
        min_ = HYPERPARAMETER_RANGES[key]['min']
        max_ = HYPERPARAMETER_RANGES[key]['max']
        normalized_hyperparameters[key] = float((val - min_))/(max_ - min_)
    return normalized_hyperparameters


def plot_radar(hyperparameters, out_path, hyperparameter_names_list):
    N = len(hyperparameters[0])
    theta = radar_factory(N, frame='polygon')
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    # ax.set_title('foobar',  position=(0.5, 1.1), ha='center')
    for hyperparameter_set in hyperparameters:
        normalized_hyperparameters = normalize_hyperparameters(hyperparameter_set)
        values = [normalized_hyperparameters[name] for name in hyperparameter_names_list]
        ax.plot(theta, values, 'ro', alpha=0.3)
    ax.set_varlabels(hyperparameter_names_list)
    plt.savefig(out_path)
    plt.close('all')


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    bayes_hyperparameters, pso_hyperparameters = collect_best_hyperparameters()
    hyperparameter_names_list = bayes_hyperparameters[0].keys()
    bayes_result, pso_result = collect_evaluations()
    plot_private_public(bayes_result, pso_result)
    plot_score_distrib(bayes_result, pso_result)
    bayes_radar_path = os.path.join(output_dir, 'bayes_radar.png')
    pso_radar_path = os.path.join(output_dir, 'pso_radar.png')
    plot_radar(bayes_hyperparameters, bayes_radar_path, hyperparameter_names_list)
    plot_radar(pso_hyperparameters, pso_radar_path, hyperparameter_names_list)
    plot_1D_hyperpar_distribs(bayes_hyperparameters, pso_hyperparameters)


def plot_score_distrib(bayes_result, pso_result):
    ###### PUBLIC ####
    plt.vlines(np.mean(pso_result['public']), ymin=0, ymax=100, colors=['g'], label='PSO mean')
    plt.vlines(np.mean(bayes_result['public']), ymin=0, ymax=100, colors=['r'], label='BO mean')
    bins = np.linspace(
        min(min(pso_result['public']), min(bayes_result['public'])),
        max(max(pso_result['public']), max(bayes_result['public'])),
        num=int(np.sqrt(len(pso_result['public'])))
    )
    plt.hist(pso_result['public'], bins=bins, color='g', label='PSO', alpha=0.5)
    plt.hist(bayes_result['public'], bins=bins, color='r', label='BO', alpha=0.5)
    plt.grid()
    plt.legend()
    plt.title('Public AMS')
    plt.xlabel('Public AMS')
    plt.ylabel('Entries per bin')
    public_out = os.path.join(output_dir, 'public.png')
    plt.savefig(public_out, bbox_inches='tight')
    plt.close('all')

    ###### PRIVATE ####
    plt.vlines(np.mean(pso_result['private']), ymin=0, ymax=100, colors=['g'], label='PSO mean')
    plt.vlines(np.mean(bayes_result['private']), ymin=0, ymax=100, colors=['r'], label='BO mean')
    bins = np.linspace(
        min(min(pso_result['private']), min(bayes_result['private'])),
        max(max(pso_result['private']), max(bayes_result['private'])),
        num=int(np.sqrt(len(pso_result['private'])))
    )
    plt.hist(pso_result['private'], bins=bins, color='g', label='PSO', alpha=0.5)
    plt.hist(bayes_result['private'], bins=bins, color='r', label='BO', alpha=0.5)
    plt.grid()
    plt.legend()
    plt.title('Private AMS')
    plt.xlabel('Private AMS')
    plt.ylabel('Entries per bin')
    private_out = os.path.join(output_dir, 'private.png')
    plt.savefig(private_out, bbox_inches='tight')
    plt.close('all')

    # Save score info
    to_save = {
        'public': {
            'Bayes': {
                'mean': np.mean(bayes_result['public']),
                'std': np.std(bayes_result['public'])
            },
            'PSO': {
                'mean': np.mean(pso_result['public']),
                'std': np.std(pso_result['public'])
            }
        },
        'private': {
            'Bayes': {
                'mean': np.mean(bayes_result['private']),
                'std': np.std(bayes_result['private'])
            },
            'PSO': {
                'mean': np.mean(pso_result['private']),
                'std': np.std(pso_result['private'])
            }
        }
    }
    output_path = os.path.join(output_dir, 'HBC_score_info.json')
    with open(output_path, 'wt') as out_file:
        json.dump(to_save, out_file)



def plot_1D_hyperpar_distribs(bayes_hyperparameters, pso_hyperparameters):
    output_dir_1D = os.path.join(output_dir, 'distrib_1D')
    if not os.path.exists(output_dir_1D):
        os.makedirs(output_dir_1D)
    hyperpar_names = bayes_hyperparameters[0].keys()
    distrib_info = {}
    for hyperparameter in hyperpar_names:
        bayes_values = [hyperparameter_set[hyperparameter] for hyperparameter_set in bayes_hyperparameters]
        pso_values = [hyperparameter_set[hyperparameter] for hyperparameter_set in pso_hyperparameters]
        bins = np.linspace(
            HYPERPARAMETER_RANGES[hyperparameter]['min'],
            HYPERPARAMETER_RANGES[hyperparameter]['max'],
            num=max(int(np.sqrt(len(bayes_values))), 10)
        )
        plt.hist(bayes_values, bins=bins, color='r', alpha=0.5, label='BO')
        plt.hist(pso_values, bins=bins, color='g', alpha=0.5, label='PSO')
        plt.vlines(np.mean(bayes_values), ymin=0, ymax=100, colors=['r'])
        plt.vlines(np.mean(pso_values), ymin=0, ymax=100, colors=['g'])
        plt.grid()
        plt.legend()
        plt.title(hyperparameter)
        plt.xlabel('Parameter value')
        plt.ylabel('Entries per bin')
        outpath = os.path.join(output_dir_1D, hyperparameter + '.png')
        plt.savefig(outpath, bbox_inches='tight')
        plt.close('all')
    # Save the distrib info
        distrib_info[hyperparameter] = {
            'Bayes': {
                'mean': np.mean(bayes_values),
                'std': np.std(bayes_values)
            },
            'PSO': {
                'mean': np.mean(pso_values),
                'std': np.std(pso_values)
            }
        }
    distrib_info_path = os.path.join(output_dir, 'hyperpar_distrib_info.json')
    with open(distrib_info_path, 'wt') as out_file:
        json.dump(distrib_info, out_file)



def plot_private_public(bayes_result, pso_result):
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    bayes_public_std_err = np.std(bayes_result['public'])/np.sqrt(len(bayes_result['public']))
    bayes_private_std_err = np.std(bayes_result['private'])/np.sqrt(len(bayes_result['public']))
    pso_public_std_err = np.std(pso_result['public'])/np.sqrt(len(pso_result['public']))
    pso_private_std_err = np.std(pso_result['private'])/np.sqrt(len(pso_result['public']))
    pso_mean = (np.mean(pso_result['private']), np.mean(pso_result['public']))
    bayes_mean = (np.mean(bayes_result['private']), np.mean(bayes_result['public']))
    plot_out = os.path.join(output_dir, 'private_public.png')
    diagonal = np.linspace(2.0, 4.0, 10)
    ax.plot(
        pso_result['private'], pso_result['public'], ls='', marker='o',
        color='g', label='PSO'
    )
    ax.plot(
        bayes_result['private'], bayes_result['public'], ls='', marker='o',
        color='r', label='BO'
    )
    ax.plot(
        diagonal, diagonal, ls='--', marker='', color='k'
    )
    bo_ellipse = Ellipse(
        bayes_mean, 2*bayes_private_std_err, 2*bayes_public_std_err,
        ls='--', color='r', alpha=0.3
    )
    pso_ellipse = Ellipse(
        pso_mean, 2*pso_private_std_err, 2*pso_public_std_err,
        ls='--', color='g', alpha=0.3
    )
    ax.plot(pso_mean[0], pso_mean[1],'k*') 
    ax.plot(bayes_mean[0], bayes_mean[1],'k*') 
    ax.add_patch(bo_ellipse)
    ax.add_patch(pso_ellipse)
    ax.legend(loc='upper left')
    plt.xlabel('Private AMS score')
    plt.ylabel('Public AMS score')
    plt.grid()
    plt.xlim(3.5, 3.75)
    plt.ylim(3.5, 3.75)
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')



def collect_evaluations():
    bayes_wcp = os.path.join(BAYES_BASE, 'higgs_submission.pso')
    pso_wcp = os.path.join(PSO_BASE, 'higgs_submission.pso')
    bayes_scores = []
    pso_scores = []
    for path in glob.glob(bayes_wcp):
        bayes_scores.append(ksc.calculate_ams_scores(
            PATH_TO_TRUTH, path
        ))
    for path in glob.glob(pso_wcp):
        pso_scores.append(ksc.calculate_ams_scores(
            PATH_TO_TRUTH, path
        ))
    pso_result = {
        'private': [i[0] for i in pso_scores],
        'public': [i[1] for i in pso_scores]
    }
    bo_result = {
        'private': [i[0] for i in bayes_scores],
        'public': [i[1] for i in bayes_scores]
    }
    return bo_result, pso_result


def collect_best_hyperparameters():
    bayes_wcp = os.path.join(BAYES_BASE, 'best_hyperparameters.json')
    pso_wcp = os.path.join(PSO_BASE, 'best_hyperparameters.json')
    bayes_hyperparameters = []
    pso_hyperparameters = []
    for path in glob.glob(bayes_wcp):
        bayes_hyperparameters.append(load_hyperparameters(path))
    for path in glob.glob(pso_wcp):
        pso_hyperparameters.append(load_pso_hyperparameters(path))
    return bayes_hyperparameters, pso_hyperparameters


def load_pso_hyperparameters(path):
    dirname = os.path.dirname(path)
    corr_hyperpar_path = os.path.join(dirname, 'corr_best_hyperparameters.json')
    submission_correction_pth = os.path.join(dirname, 'higgs_submission.pso')
    if not os.path.exists(corr_hyperpar_path) or not os.path.exists(submission_correction_pth):
        wcp = os.path.join(dirname, 'previous_files', 'iteration_*', '*', 'score.json')
        max_ = 0
        for test_path in glob.glob(wcp):
            with open(test_path, 'rt') as in_file:
                score = json.load(in_file)['d_ams']
                score *= -1
                if score > max_:
                    max_ = score
                    pth = test_path
        best_dirname = os.path.dirname(pth)
        best_hyperparameters_pth = os.path.join(best_dirname, 'parameters.json')
        with open(best_hyperparameters_pth, 'rt') as in_file:
            best_hyperparameters = json.load(in_file)
        seed = int(best_dirname.split('/')[-4])
        if not os.path.exists(submission_correction_pth):
            sh.submission_creation(
                PATH_TO_TRAIN,
                PATH_TO_TEST,
                best_hyperparameters,
                submission_correction_pth,
                seed=seed
            )
        if not os.path.exists(corr_hyperpar_path):
            with open(corr_hyperpar_path, 'w') as out_file:
                json.dump(best_hyperparameters, out_file)
    else:
        with open(corr_hyperpar_path, 'rt') as in_file:
            best_hyperparameters = json.load(in_file)
    return best_hyperparameters


def load_hyperparameters(path):
    with open(path, 'rt') as in_file:
        hyperparameters = json.load(in_file)
    return hyperparameters


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super(RadarAxes, self).__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, closed=True, *args, **kwargs):
            """Override fill so that line is closed by default"""
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super(RadarAxes, self).draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super(RadarAxes, self)._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        main()
    except docopt.DocoptExit as e:
        print(e)