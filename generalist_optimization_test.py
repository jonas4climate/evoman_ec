import random
import numpy as np
import os
import pandas as pd
import tqdm
import multiprocessing

import optuna
from optuna.samplers import TPESampler

from deap import base, creator, tools, cma
from evoman.environment import Environment
from controller_cmaes import controller_cmaes

# Two groups of enemies
ENEMY_SET_1 = [3, 5, 7]
ENEMY_SET_2 = [2, 6, 7, 8]

# Folders to be created for data storage
EXP_NAME_1, EXP_NAME_2 = f'{ENEMY_SET_1}', f'{ENEMY_SET_2}'
DATA_FOLDER_1, DATA_FOLDER_2 = os.path.join('data', 'cmaes', EXP_NAME_1), os.path.join('data', 'cmaes', EXP_NAME_2)

# Number of repeated runs
N_RUNS = 2

# Hall-of-fame size (number of best individuals during evolution saved)
HOF_SIZE = 1

## Hyperparameters for CMA-ES
POPULATION_SIZE = 100
SIGMA = 2.5
NGEN = 10

## Hyperparameter search parameters
HP_POP_SIZE_RANGE = (10, 500)
HP_SIGMA_RANGE = (0.1, 10.0)
HP_N_RUNS = 2
HP_N_TRIALS = 10
HP_PARALLEL_RUNS = os.cpu_count()  # Will control how many processes we spawn

HP_FITNESS_CRITERION = np.max  # Fitness criterion
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

def run_evolutions(env, config, n_runs=1, pbar_pos=2):
    pbar_gens = tqdm.tqdm(total=n_runs*NGEN, desc=f'Run {pbar_pos-1} | sigma={config.sigma:.2f}  n_pop={config.population_size}', unit='gen', position=pbar_pos)
    N = 21 * NUM_HIDDEN + (NUM_HIDDEN + 1) * 5
    all_fitnesses = np.zeros((n_runs, NGEN, config.population_size))
    best_individuals = np.zeros((n_runs, N))

    for run in range(n_runs):
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
            all_fitnesses[run, gen] = [ind.fitness.values[0] for ind in population]
            pbar_gens.update(1)

        best_individuals[run] = hof[0]
    
    df_stats = pd.DataFrame(logbook)
    pbar_gens.close()
    return all_fitnesses, best_individuals, df_stats

def run_trial_in_subprocess(trial, conn, config):
    # trial_seed = SEED + trial.number
    # np.random.seed(trial_seed)
    # random.seed(trial_seed)

    env = create_environment(EXP_NAME_1, ENEMY_SET_1)
    
    all_fitnesses, _, _ = run_evolutions(env, config, HP_N_RUNS, pbar_pos=2+trial.number)
    hp_fitness = float(HP_FITNESS_CRITERION(all_fitnesses))
    if not isinstance(hp_fitness, (float, int)) or hp_fitness is None:
        raise ValueError(f"Invalid fitness value for trial {trial.number}: {hp_fitness}")

    conn.send(hp_fitness)
    conn.close()

def hyperparameter_search():
    # pbar = tqdm.tqdm(total=HP_N_TRIALS, desc='Hyperparameter search', unit='trial', position=1, leave=True)

    def run_trial(trial):
        sigma = trial.suggest_float('sigma', *HP_SIGMA_RANGE)
        pop_size = trial.suggest_int('pop_size', *HP_POP_SIZE_RANGE, log=True)
        config = CMAESConfig(sigma=sigma, population_size=pop_size)
        parent_conn, child_conn = multiprocessing.Pipe()
        process = multiprocessing.Process(target=run_trial_in_subprocess, args=(trial, child_conn, config))
        process.start()
        hp_fitness = parent_conn.recv()  # Get the fitness result from the child process
        process.join()
        if hp_fitness is None:
            raise RuntimeError(f"Trial {trial.number} failed.")
        return hp_fitness
    
    # def update_pbar(study, trial):
    #     pbar.update(1)

    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    # study.optimize(run_trial, n_trials=HP_N_TRIALS, n_jobs=HP_PARALLEL_RUNS, callbacks=[update_pbar])
    study.optimize(run_trial, n_trials=HP_N_TRIALS, n_jobs=HP_PARALLEL_RUNS)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(DATA_FOLDER_1, 'hp_trials_data.csv'), index=False)

    best_params = study.best_params
    sigma = best_params['sigma']
    population_size = best_params['pop_size']
    config = CMAESConfig(sigma=sigma, population_size=population_size)
    df = pd.DataFrame(best_params, index=[0])
    df.to_csv(os.path.join(DATA_FOLDER_1, 'hp_best_params.csv'), index=False)

    return config

if __name__ == '__main__':
    hyperparameter_search()