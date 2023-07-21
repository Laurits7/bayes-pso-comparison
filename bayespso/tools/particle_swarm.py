import numpy as np
import os
import json
import glob
import chaospy as cp


class Particle():

    def __init__(self, hyperparameter_info, hyperparameters, iterations):
        self.confidence_coefficients = {'c_max': 1.62, 'w': 0.8, 'w2': 0.4}
        self.set_inertial_weight_step(iterations)
        self.hyperparameter_info = hyperparameter_info
        self.names = hyperparameter_info.keys()
        self.hyperparameters = hyperparameters
        self.initialize_speeds()
        self.personal_best_history = []
        self.personal_best_fitness_history = []
        self.fitness_history = []
        self.location_history = []
        self.total_iterations = iterations
        self.iteration = 0

    def set_inertial_weight_step(self, iterations):
        range_size = (
            self.confidence_coefficients['w'] - \
            self.confidence_coefficients['w2']
        )
        self.weight_step = range_size / iterations

    def initialize_speeds(self):
        self.speed = {}
        for key in self.names:
            v_max = (
                self.hyperparameter_info[key]['max'] - \
                self.hyperparameter_info[key]['min'] / 4
            )
            self.speed[key] = np.random.uniform() * v_max

    def set_fitness(self, fitness):
        self.fitness = fitness
        if self.fitness < self.personal_best_fitness:
            self.set_personal_best()
        if self.fitness < self.global_best_fitness:
            self.set_global_best(self.hyperparameters, self.fitness)

    def set_personal_best(self):
        self.personal_best = self.hyperparameters.copy()
        self.personal_best_fitness = float(self.fitness)

    def set_global_best(self, hyperparameters, fitness):
        self.global_best = hyperparameters.copy()
        self.global_best_fitness = float(fitness)


    def set_initial_bests(self, fitness):
        self.fitness = fitness
        self.set_personal_best()
        self.set_global_best(self.hyperparameters, self.fitness)

    def update_speeds(self):
        for key in self.names:
            rand1 = np.random.uniform()
            rand2 = np.random.uniform()
            cognitive_component = self.confidence_coefficients['c_max'] * rand1 * (
                self.personal_best[key] - self.hyperparameters[key])
            social_component = self.confidence_coefficients['c_max'] * rand2 * (
                self.global_best[key] - self.hyperparameters[key])
            inertial_component = (
                self.confidence_coefficients['w'] * self.speed[key]
            )
            self.speed[key] = (
                cognitive_component
                + social_component
                + inertial_component
            )

    def update_location(self):
        for key in self.names:
            self.hyperparameters[key] += self.speed[key]
            if self.hyperparameter_info[key]['exp'] == 1:
                max_value = np.exp(self.hyperparameter_info[key]['max'])
                min_value = np.exp(self.hyperparameter_info[key]['min'])
            else:
                max_value = self.hyperparameter_info[key]['max']
                min_value = self.hyperparameter_info[key]['min']
            if self.hyperparameters[key] > max_value:
                self.hyperparameters[key] = max_value
                self.speed[key] = 0
            if self.hyperparameters[key] < min_value:
                self.hyperparameters[key] = min_value
                self.speed[key] = 0
            if self.hyperparameter_info[key]['int'] == 1:
                self.hyperparameters[key] = int(np.ceil(self.hyperparameters[key]))

    def gather_intelligence(self, locations, fitnesses):
        index = np.argmin(fitnesses)
        min_fitness = min(fitnesses)
        if min_fitness < self.global_best_fitness:
            self.set_global_best(locations[index], fitnesses[index])

    def track_history(self):
        self.personal_best_history.append(self.personal_best)
        self.personal_best_fitness_history.append(self.personal_best_fitness)
        self.fitness_history.append(self.fitness)
        self.location_history.append(self.hyperparameters)

    def next_iteration(self):
        self.update_location()
        self.update_speeds()
        self.track_history()
        self.confidence_coefficients['w'] -= self.weight_step


class ParticleSwarm:
    def __init__(self, settings, fitness_function, hyperparameter_info):
        self.settings = settings
        self.fitness_function = fitness_function
        self.output_dir = settings['output_dir']
        self.hyperparameter_info = hyperparameter_info
        self.names = list(hyperparameter_info.keys())
        self.global_bests = []
        self.global_best = 99e99
        self.swarm = self.createSwarm()

    def createSwarm(self):
        particle_swarm = []
        locations = self.create_initial_locations()
        for location in locations:
            single_particle = Particle(
                self.hyperparameter_info,
                location,
                self.settings['iterations']
            )
            particle_swarm.append(single_particle)
        return particle_swarm

    def create_initial_locations(self):
        list_of_distributions = []
        for name in self.names:
            hyperparameter = self.hyperparameter_info[name]
            list_of_distributions.append(
                cp.Uniform(hyperparameter['min'], hyperparameter['max'])
            )
        distribution = cp.J(*list_of_distributions)
        samples = distribution.sample(self.settings['sample_size'], rule='latin_hypercube')
        sample_points = np.transpose(samples)
        locations = []
        for sample_point in sample_points:
            location = {}
            for coord, name in zip(sample_point, self.names):
                if self.hyperparameter_info[name]['int']:
                     value = np.round(coord).astype(int)
                elif self.hyperparameter_info[name]['exp']:
                    value = np.exp(coord)
                else:
                    value = coord
                location[name] = value
            locations.append(location)
        return locations

    def espionage(self):
        for particle in self.swarm:
            informants = np.random.choice(
                self.swarm, self.settings['nr_informants']
            )
            best_fitnesses, best_locations = self.get_fitnesses_and_location(
                informants)
            particle.gather_intelligence(best_locations, best_fitnesses)

    def get_fitnesses_and_location(self, group):
        best_locations = []
        best_fitnesses = []
        for particle in group:
            best_fitnesses.append(particle.personal_best_fitness)
            best_locations.append(particle.personal_best)
        return best_fitnesses, best_locations

    def set_particle_fitnesses(self, fitnesses, initial=False):
        for particle, fitness in zip(self.swarm, fitnesses):
            if initial:
                particle.set_initial_bests(fitness)
            else:
                particle.set_fitness(fitness)

    def find_best_hyperparameters(self):
        best_fitnesses, best_locations = self.get_fitnesses_and_location(
            self.swarm)
        index = np.argmin(best_fitnesses)
        best_fitness = best_fitnesses[index]
        best_location = best_locations[index]
        return best_fitness, best_location

    def check_global_best(self):
        for particle in self.swarm:
            if particle.fitness < self.global_best:
                self.global_best = particle.fitness
        self.global_bests.append(self.global_best)

    def particleSwarmOptimization(self):
        iteration = 0
        np.random.seed(self.settings['seed'])
        if True:
            last_complete_iteration = collect_iteration_particles(self.output_dir)
            fitnesses, all_locations = get_iteration_info(
                self.output_dir, iteration, self.settings)
        else:
            all_locations = [particle.hyperparameters for particle in self.swarm]
            fitnesses = self.fitness_function(all_locations, self.settings)
        self.set_particle_fitnesses(fitnesses, initial=True)
        self.check_global_best()
        for particle in self.swarm:
            particle.next_iteration()
        self.save_iter_info(iteration, self.settings['output_dir'])
        self.set_particle_fitnesses(fitnesses, initial=True)
        for particle in self.swarm:
            particle.next_iteration()
        iteration = 1
        while iteration < self.settings['iterations']:
            print('%s/%s' %(iteration, self.settings['iterations']))
            iteration += 1
            self.espionage()
            all_locations = [particle.hyperparameters for particle in self.swarm]
            if self.settings['continue'] and iteration <= last_complete_iteration:
                fitnesses, all_locations = get_iteration_info(
                    self.output_dir, iteration, self.settings)
            else:
                fitnesses = self.fitness_function(all_locations, self.settings)
            self.set_particle_fitnesses(fitnesses)
            self.check_global_best()
            self.save_iter_info(iteration, self.settings['output_dir'])
            for particle in self.swarm:
                particle.next_iteration()
        best_fitness, best_location = self.find_best_hyperparameters()
        return best_location, best_fitness

    def save_iter_info(self, iteration, output_dir):
        best_fitness = self.global_best
        output_path = os.path.join(output_dir, 'evol.json')
        d = {'iteration': iteration, 'best_fitness': best_fitness}
        with open(output_path, 'at') as info_file:
            json.dump(d, info_file)
            info_file.write('\n')


def collect_iteration_particles(iteration_dir):
    iteration_paths = os.path.join(
        iteration_dir, 'previous_files', 'iteration_*')
    all_iterations = glob.glob(iteration_paths)
    return check_last_iteration_completeness(all_iterations, iteration_dir)


def check_last_iteration_completeness(all_iterations, iteration_dir):
    iteration_nrs = [
        int(iteration.split('_')[-1]) for iteration in all_iterations
    ]
    iteration_nrs.sort()
    last_iteration = os.path.join(
        iteration_dir, 'iteration_' + str(iteration_nrs[-1]))
    all_particles_wildcard = os.path.join(last_iteration, '*')
    for path in glob.glob(all_particles_wildcard):
        parameter_file = os.path.join(path, 'parameters.json')
        score_file = os.path.join(path, 'score.json')
        if not os.path.exists(parameter_file):
            return iteration_nrs[-2]
        if not os.path.exists(score_file):
            return iteration_nrs[-2]
    return iteration_nrs[-1]


def get_iteration_info(output_dir, iteration, settings):
    number_particles = settings['sample_size']
    iteration_dir = os.path.join(
        output_dir, 'previous_files', 'iteration_' + str(iteration))
    fitnesses = []
    parameters_list = []
    for particle in range(number_particles):
        particle_dir = os.path.join(iteration_dir, str(particle))
        score_file = os.path.join(particle_dir, 'score.json')
        parameter_file = os.path.join(particle_dir, 'parameters.json')
        with open(score_file, 'rt') as inFile:
            fitness = json.load(inFile)['d_ams']
        with open(parameter_file, 'rt') as inFile:
            parameters = json.load(inFile)
        fitnesses.append(fitness)
        parameters_list.append(parameters)
    return fitnesses, parameters_list
