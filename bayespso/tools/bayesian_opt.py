from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
from heapq import nsmallest
import time
import json
import os
import chaospy as cp
import os
import glob


class BayesianOptimizer(object):
    """ Bayesian optimizer class for optimizing using Gaussian processes with
    the expected improvement as the acquisition function """
    def __init__(
        self, hyperparameters, objective, global_settings, niter=500,
        nparallel_eval=1, npoints_init=100, exploration=0.01, sample_points=1000
    ):
        """ Initializes the BayesianOptimizer class with the set parameters

        Args:
            hyperparameters : dict
                Dictionary containing the hyperparameters and the minimum and
                maximum allowed values. Should be of the format
                hyperparameters = {'param1': {'min': xyz, 'max': xyz}}
            objective : function
                Objective function to be optimized with the Bayesian optimization.
                The interface of the function should accept the hyperparameters
                as a dictionary as such: [{'a': 1, 'b': 2}]
            [niter=500] : int
                Number of iterations the optimizer will run
            [nparallel_eval=1] : int
                Number of evaluations to be done in parallel
            [npoints_init=100] : int
                Number of initial points to be used
            [exploration=0.3] : float
                Constatnt that influences the exploration-exploitation trade-off.
            [sample_points=1000] : int
                Number of points to be sampled by the aqcuisition function
        """
        self.hyperparameters = hyperparameters
        self.hyperpar_list = [parameter for parameter in hyperparameters]
        self.objective = objective
        self.names = list(self.hyperparameters.keys())
        self.global_settings = global_settings
        self.niter = niter
        self.nparallel_eval = nparallel_eval
        self.npoints_init = npoints_init
        self.exploration = exploration
        self.sample_points = sample_points
        self.dimension = len(hyperparameters)
        # self.exploration_step = self.calculate_exploration_step()

    def calculate_exploration_step(self):
        return self.exploration / self.niter

    def generate_random_positions(self, number_points):
        """ Generates a list of random positions according to a sampling
        algorithm

        Args:
            number_points : int
                Number of random points to be generated

        Returns:
            random_positions_l : numpy.array
                Array of random positions
            random_positions_d : list of dicts
                List of random positions
        """
        
        list_of_distributions = []
        for name in self.names:
            hyperparameter = self.hyperparameters[name]
            list_of_distributions.append(
                cp.Uniform(hyperparameter['min'], hyperparameter['max'])
            )
        distribution = cp.J(*list_of_distributions)
        samples = distribution.sample(number_points, rule='latin_hypercube')
        sample_points = np.transpose(samples)
        sample_locations_d = []
        sample_locations_l = []
        for location in sample_points:
            sample_location_d = {}
            sample_location_l = []
            for coord, name in zip(location, self.names):
                if self.hyperparameters[name]['int']:
                     value = np.round(coord).astype(int)
                elif self.hyperparameters[name]['exp']:
                    value = np.exp(coord)
                else:
                    value = coord
                sample_location_d[name] = value
                sample_location_l.append(value)
            sample_locations_d.append(sample_location_d)
            sample_locations_l.append(sample_location_l)
        return np.array(sample_locations_l), sample_locations_d

    def generate_initial_points(self):
        """ Generates initial points and evaluates them """
        initial_positions_l, initial_positions_d = self.generate_random_positions(self.npoints_init)
        initial_function_values = np.array(self.objective(initial_positions_d, self.global_settings))
        return initial_positions_l, initial_function_values.reshape(-1, 1)

    def save_results(self, all_values, positions, function_values, iteration):
        output_path = os.path.join(
            self.global_settings['output_dir'],
            'evol.json'
        )
        function_values = list([list(fun) for fun in function_values])
        with open(output_path, 'at') as out_file:
            d = {
                'iteration': iteration,
                'best_value': min(all_values)[0],
                'position': positions,
                'function_values': function_values
            }
            json.dump(d, out_file)
            out_file.write('\n')

    def save_timing(self, iteration, time_spent):
        output_path = os.path.join(
            self.global_settings['output_dir'], 'time.json'
        )
        with open(output_path, 'at') as out_file:
            d = {'iteration': iteration, 'time': time_spent}
            json.dump(d, out_file)
            out_file.write('\n')

    def expected_improvement(self, sample, mu_sample_opt, model):
        """
        Computes the EI at points X based on existing samples XY
        using a Gaussian process surrogate model. Not using the actual function
        values of the probed location because we are assuming a noisy model.

        Args:
            sample: Points at which EI shall be computed (m x d). (randomly generated)
            XY: Sample locations (n x d). (Probed with objective unction)


        Returns:
            Expected improvements at points "sample".
        """
        mu, sigma = model.predict(sample, return_std=True)
        # mu_sample = model.predict(XY)
        # mu_sample_opt = np.min(mu_sample)
        sigma = sigma.reshape(-1, 1)
        with np.errstate(divide='warn'):
            # https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf
            # imp = mu - mu_sample_opt - self.exploration ---> should be max(0, f' - f(x))
            # so if f(x) is smaller, then there is improvement
            # In our case this corresponds to max(0, mu_sample_opt - mu - self.exploration)
            imp = mu_sample_opt - mu - self.exploration
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return float(ei)

    def propose_single_location(
            self, best_function_value, model, n_restarts=5
    ):
        """
        Proposes the next sampling point by optimizing the acquisition function.
        The minimization is run n_restarts times to avoid getting stuck at a
        local minimum

        Args:
            best_function_value: The already probed locations.
            model: A GaussianProcessRegressor fitted to samples.

        Returns:
            Location of the acquisition function maximum.
        """
        random_positions_l, random_positions_d = self.generate_random_positions(n_restarts)
        dim = random_positions_l.shape[1]
        best_value = 0
        bounds = []
        for name in self.names:
            tmp_min = self.hyperparameters[name]['min']
            min_bound = np.exp(tmp_min) if self.hyperparameters[name]['exp'] else tmp_min
            tmp_max = self.hyperparameters[name]['max']
            max_bound = np.exp(tmp_max) if self.hyperparameters[name]['exp'] else tmp_max
            bounds.append((min_bound, max_bound))
        def min_obj(X):
            ei = self.expected_improvement(X.reshape(-1, dim), best_function_value, model)
            # Since we want to maximize EI, then we should minimize negative EI
            return -float(ei)
        for location in random_positions_l:
            res = minimize(min_obj, x0=location, bounds=bounds, method='L-BFGS-B')
            if res.fun < best_value:
                best_value = res.fun
                best_location_l = res.x
        best_loc_l = []
        for name, value in zip(self.names, best_location_l):
            if self.hyperparameters[name]['int']:
                value = int(np.round(value))
            best_loc_l.append(value)
        best_location_d = {name: value for name, value in zip(self.names, best_loc_l)}
        return best_loc_l, best_location_d

    def propose_location(self, positions, function_values, model):
        ''' qEI version '''
        best_function_value = np.array([np.min(function_values)]).reshape(-1, 1)
        new_probes_l = []
        new_probes_d = []
        for i in range(self.nparallel_eval):
            best_location_l, best_location_d = self.propose_single_location(
                best_function_value, model
            )
            positions = np.vstack((positions, np.array(best_location_l).reshape(1, self.dimension)))
            function_values = np.vstack((function_values, best_function_value))
            model.fit(positions, function_values)
            new_probes_l.append(best_location_l)
            new_probes_d.append(best_location_d)
        return new_probes_l, new_probes_d

    def save_final_result(self, best_loc, best_value):
        output_path = os.path.join(self.global_settings['output_dir'], 'final_result.json')
        with open(output_path, 'at') as out_file:
            d = {'best_loc': best_loc, 'function_value': best_value}
            json.dump(d, out_file)

    def continue_previous_run(self):
        ''' Continues the previous run that was interrupted'''
        continuation_path = os.path.join(
            self.global_settings['output_dir'], 'evol.json')
        positions_d = []
        positions_l = []
        function_values = []
        with open(continuation_path, 'rt') as inFile:
            for line in inFile:
                d = json.loads(line)
                function_values.extend(d['function_values'])
                for iter_position in d['position']:
                    position_l = [iter_position[name] for name in self.names]
                    positions_l.append(position_l)
                positions_d.extend(d['position'])
        return positions_l, function_values

    def collect_initialization(self):
        initial_positions = os.path.join(
            self.global_settings['output_dir'], 'previous_files', 'iteration_0')
        positions = []
        function_values = []
        nr_initial_positions = len(glob.glob(os.path.join(initial_positions, '*')))
        for p in range(nr_initial_positions):
            initial_pos_dir = os.path.join(initial_positions, str(p))
            param_file = os.path.join(initial_pos_dir, 'parameters.json')
            score_file = os.path.join(initial_pos_dir, 'score.json')
            with open(param_file, 'rt') as inFile:
                param = json.load(inFile)
                positions.append([param[name] for name in self.names])
            with open(score_file, 'rt') as inFile:
                score = json.load(inFile)['d_ams']
                function_values.append(score)
        return positions, np.array(function_values).reshape(-1, 1)

    def optimize(self):
        """ Finds the best hyperparameters for a given objective function

        Returns:
            best_hyperparameters : dict
                Best found hyperparameters
            best_function_value : float
                Function value at the best found hyperparameters
        """
        np.random.seed(self.global_settings['seed'])
        try:
            if 'HBC' in self.global_settings['output_dir']:
                positions, function_values = self.collect_initialization()
                new_positions, new_function_values = self.continue_previous_run()
                positions = np.vstack((positions, new_positions))
                function_values = np.vstack((function_values, new_function_values))
                init_iter = int(len(positions)/self.nparallel_eval) - 1
                print("Starting from iteration: ", init_iter)
            else:
                positions, function_values = self.continue_previous_run()
                init_iter = int(len(positions)/self.nparallel_eval)
        except FileNotFoundError:# FileNotFoundError is in py3 and IOError in py2
            positions, function_values = self.generate_initial_points()
            init_iter = 0
        customKernel = C(1.0, (0.01, 1000.0)) * Matern(
            length_scale=np.ones(self.dimension),
            length_scale_bounds=[(0.01, 100)] * self.dimension, nu=2.5
        ) + WhiteKernel()
        model = GaussianProcessRegressor(
            kernel=customKernel,
            n_restarts_optimizer=2,
            normalize_y=True
        )
        for i in range(init_iter, self.niter):
            print("Iteration %s/%s" %(i + 1, self.niter))
            start = time.time()
            model.fit(positions, function_values)
            new_positions, new_positions_d = self.propose_location(
                positions, function_values, model
            )
            new_function_values = np.array([self.objective(new_positions_d, self.global_settings)]).reshape(-1, 1)
            positions = np.vstack((positions, new_positions))
            function_values = np.vstack((function_values, new_function_values))
            # self.exploration -= self.exploration_step
            end = time.time()
            self.save_timing(i, end-start)
            self.save_results(function_values, new_positions_d, new_function_values, i)
        index = np.argmin(function_values)
        print("Function value: "+ str(function_values[index]))
        print("Position: " + str(positions[index]))
        self.save_final_result(list(positions[index]), list(function_values[index]))
        return positions[index], function_values[index]
