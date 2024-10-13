import random
import os
import sys
import multiprocessing
import tqdm
import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler

from deap import base, creator, tools, cma
from evoman.environment import Environment
from controller_cmaes import controller_cmaes
from util import crit_mean_of_max, crit_max, crit_mean

# Two groups of enemies
ENEMY_SETS = {
    'set_1': [3, 5, 7],
    'set_2': [2, 6, 7, 8]
}

# Number of repeated runs
N_RUNS = 5

# Hall-of-fame size (number of best individuals during evolution saved)
HOF_SIZE = 1

## Hyperparameters for CMA-ES
POPULATION_SIZE = 100 # (tune-able)
SIGMA = 2.5 # (tune-able)
NGEN = 200 # (not tune-able)

## Hyperparameter search parameters
HP_POP_SIZE_RANGE = (10, 200)
HP_SIGMA_RANGE = (0.1, 10.0)
HP_N_RUNS = 3
HP_N_TRIALS = 10
HP_PARALLEL_RUNS = 6 # os.cpu_count()  # 1 for serial implementation

HP_FITNESS_CRITERION = crit_mean_of_max # Alternatives defined in util.py
NUM_HIDDEN = 10  # Number of hidden layers

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)

class CMAESConfig():
    def __init__(self, sigma, population_size):
        self.sigma = sigma
        self.population_size = population_size

    def __str__(self):
        return f"CMA-ES (sigma={self.sigma}, population_size={self.population_size})"

    def __repr__(self):
        return self.__str__()

def evaluate_fitness(individual, environment):
    environment.player_controller.set_weights(individual, NUM_HIDDEN)
    agg_fitness, _, _, _ = environment.play()
    return agg_fitness,

def create_environment(experiment_name, enemy_set, visuals=False):
    return Environment(experiment_name=experiment_name,
                        enemies=enemy_set,
                        multiplemode="yes",
                        enemymode='static',
                        speed='normal' if visuals else 'fastest',
                        player_controller=controller_cmaes(),
                        savelogs="no",
                        logs="off",
                        clockprec="low",
                        visuals=visuals)

def setup_toolbox(N, env, config):
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fitness, environment=env)
    centroid = np.zeros(N)
    strategy = cma.Strategy(centroid=centroid, sigma=config.sigma, lambda_=config.population_size)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    return toolbox

def setup_data_collector():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats


def run_evolutions(env, config, n_runs=1, pbar_pos=2, parallel=False):
    N = 21 * NUM_HIDDEN + (NUM_HIDDEN + 1) * 5
    all_fitnesses = np.zeros((n_runs, NGEN, config.population_size))
    best_individuals = np.zeros((n_runs, N))
    list_run_stats = []

    def run_evolution(run, env, config):
        pbar_gens = tqdm.tqdm(total=NGEN, desc=f'Run {run+1} | sigma={config.sigma:.2f}  n_pop={config.population_size:3d}', unit='gen', position=pbar_pos+run)

        # Custom seeding
        run_seed = SEED + HP_N_TRIALS + run + 1
        np.random.seed(run_seed)
        random.seed(run_seed)

        run_fitnesses = np.zeros((NGEN, config.population_size))

        toolbox = setup_toolbox(N, env, config)
        population = toolbox.population(n=config.population_size)
        hof = tools.HallOfFame(HOF_SIZE, similar=lambda a, b: np.array_equal(a.fitness.values, b.fitness.values))
        stats = setup_data_collector()
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        for gen in range(NGEN):
            population = toolbox.generate()
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            toolbox.update(population)
            hof.update(population)
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(population), **record)
            run_fitnesses[gen] = [ind.fitness.values[0] for ind in population]
            pbar_gens.update(1)

        best_individual = hof[0]
        run_stats = pd.DataFrame(logbook)
        return run_fitnesses, best_individual, run_stats

    if parallel:
        raise NotImplementedError("Haven't figured this out yet.")
    else:
        pbar_runs = tqdm.tqdm(total=n_runs, desc='Runs', unit='run', position=pbar_pos)
        for run in range(n_runs):
            run_fitnesses, best_individual, run_stats = run_evolution(run, env, config)
            all_fitnesses[run] = run_fitnesses
            best_individuals[run] = best_individual
            list_run_stats.append(run_stats)
            pbar_runs.update(1)

    return all_fitnesses, best_individuals, list_run_stats

def run_trial_in_subprocess(trial, conn, config, set_name, enemy_set, f_criterion=HP_FITNESS_CRITERION):
    # Custom seeding
    trial_seed = SEED + trial.number
    np.random.seed(trial_seed)
    random.seed(trial_seed)

    env = create_environment(set_name, enemy_set)
    
    all_fitnesses, _, _ = run_evolutions(env, config, HP_N_RUNS, pbar_pos=2+trial.number)
    hp_fitness = float(f_criterion(all_fitnesses))

    if not isinstance(hp_fitness, (float, int)) or hp_fitness is None:
        raise ValueError(f"Invalid fitness value for trial {trial.number}: {hp_fitness}")
    
    if conn is not None:
        conn.send(hp_fitness)
        conn.close()

def get_best_weights(all_fitnesses, best_individuals):
    best_run_idx = all_fitnesses.max(axis=(1, 2)).argmax()
    best_individual = best_individuals[best_run_idx]
    return best_individual

def hyperparameter_search(set_name, data_folder, enemy_set):
    # pbar = tqdm.tqdm(total=HP_N_TRIALS, desc='Hyperparameter search', unit='trial', position=1, leave=True)

    def run_trial(trial, parallel=False):
        sigma = trial.suggest_float('sigma', *HP_SIGMA_RANGE, log=True)
        pop_size = trial.suggest_int('pop_size', *HP_POP_SIZE_RANGE)
        config = CMAESConfig(sigma=sigma, population_size=pop_size)
        if parallel:
            parent_conn, child_conn = multiprocessing.Pipe()
            process = multiprocessing.Process(target=run_trial_in_subprocess, args=(trial, child_conn, config, set_name, enemy_set))
            process.start()
            hp_fitness = parent_conn.recv()
            process.join()

            process.close()
            parent_conn.close()
            child_conn.close()
        else:
            hp_fitness = run_trial_in_subprocess(trial, None, config, set_name, enemy_set)

        if hp_fitness is None:
            raise RuntimeError(f"Trial {trial.number} failed.")
        return hp_fitness
    
    crit_name = HP_FITNESS_CRITERION.__name__
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    parallel = True if HP_PARALLEL_RUNS > 1 else False
    study.optimize(lambda trial: run_trial(trial, parallel), n_trials=HP_N_TRIALS, n_jobs=HP_PARALLEL_RUNS, gc_after_trial=True)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(data_folder, f'hp_{crit_name}_trials_data.csv'), index=False)

    best_params = study.best_params
    sigma = best_params['sigma']
    population_size = best_params['pop_size']
    config = CMAESConfig(sigma=sigma, population_size=population_size)
    df = pd.DataFrame(best_params, index=[0])
    df.to_csv(os.path.join(data_folder, f'hp_{crit_name}_best_params.csv'), index=False)

    return config

if __name__ == '__main__':
    # Setup
    set_names, enemy_sets = zip(*ENEMY_SETS.items())
    data_folders = [os.path.join('data', 'cmaes', f'{set_name}', f'hp_{HP_FITNESS_CRITERION.__name__}') for set_name in set_names]
    for folder in data_folders:
        os.makedirs(folder, exist_ok=True)
    
    # Core
    for name, folder, enemy_set in zip(set_names, data_folders, enemy_sets):
        if '--tune' in sys.argv:
            print('Generating hyperparameter values through tuning...')
            config = hyperparameter_search(name, folder, enemy_set)
        elif '--load' in sys.argv:
            print('Loading hyperparameter values from file...')
            try:
                with open(os.path.join(folder, f'hp_best_params.csv'), 'r') as f:
                    df = pd.read_csv(f)
                    config = CMAESConfig(sigma=df['sigma'].values[0], population_size=df['pop_size'].values[0])
            except FileNotFoundError:
                print(f"File not found. You likely have not run tuning with {HP_FITNESS_CRITERION.__name__}.")
                sys.exit(1)
        else:
            print('Using hard-coded hyperparameter values in script.')
            config = CMAESConfig(sigma=SIGMA, population_size=POPULATION_SIZE)

        print(config)

        if '--train' in sys.argv:
            print(f'Training the ideal controller from {N_RUNS} evolutions...')
            env = create_environment(name, enemy_set)
            all_fitnesses, best_individuals, list_df_stats = run_evolutions(env, config, n_runs=N_RUNS)
            np.save(os.path.join(folder, 'train_all_fitnesses.npy'), all_fitnesses)
            np.save(os.path.join(folder, 'train_best_individuals.npy'), best_individuals)
            for run, best_individual in enumerate(best_individuals):
                with open(os.path.join(folder, f"train_stats_run{run + 1}.csv"), 'w') as f:
                    list_df_stats[run].to_csv(f, index=False)

        if '--show' in sys.argv:
            print('Showcasing the best generalist...')
            env = create_environment(name, enemy_set, visuals=True)
            env.speed = 'normal'
            all_fitnesses = np.load(os.path.join(folder, 'train_all_fitnesses.npy'))
            best_individuals = np.load(os.path.join(folder, 'train_best_individuals.npy'))
            best_weights = get_best_weights(all_fitnesses, best_individuals)
            env.player_controller.set_weights(best_weights, NUM_HIDDEN)
            agg_fit, _, _, _ = env.play()
            print(f"Fit = {agg_fit}")
