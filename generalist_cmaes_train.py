import random
import os
import sys
import tqdm
import numpy as np
import pandas as pd

from deap import base, creator, tools, cma
from generalist_shared import create_environment

# Use configuration unless specified otherwise
from generalist_cmaes_config import *

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

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
    """Helper function for evaluating fitness of a given individual.
    Args:
        individual (np.array[float]): array of weights to be fed into the player's neural network
        environment (Environment): Evoman framework's Environment object to run the game on
    Returns:
        agg_fitness: Fitness aggregated from all enemies the game was ran on
    """    
    # Set the weights corresponding to the given individual
    environment.player_controller.set_weights(individual, NUM_HIDDEN)

    # Run the game against all opponents and return default aggregated fitness
    agg_fitness, p_life, e_life, time = environment.play()
    return agg_fitness,


def setup_toolbox(N, env, config):
    """Helper function for setting up toolbox (setup) from the Deap package.
    Args:
        N (int): no. of weights required for the neural network
        env (Environment): Environment object representing the the Evoman instance
        config (CMAESConfig): configuration object for CMA-ES
    Returns:
        toolbox: Deap's toolbox object populated with all necessary functions for CMA-ES
    """
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
    """Helper function for setting up data collector from the Deap package.
    Returns:
        stats: Deap's statistics object with basic statics for simple inspection post-evolution
    """
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats


def run_evolutions(env, config, n_runs=1, pbar_pos=2, parallel=False):
    """Wrapper function for running multiple evolutions of CMA-ES on the given environment and collecting data.
    Args:
        env (Environment): Environment object representing the the Evoman instance
        config (CMAESConfig): configuration object for CMA-ES
        n_runs (int, optional): number of repeated runs.
        pbar_pos (int, optional): position of the progress bar.
        parallel (bool, optional): Whether to run the evolution across runs in parallel. Not yet implemented.
    Returns:
        all_fitnesses: 3D array with fitness values for each individual in each generation in each run
        best_individuals: 2D array with best individual from each run
        list_run_stats: list of dataframes with statistics for each generation in each run
    """
    # Numer of weights in the neural network: 21 inputs (no. of sensors + bias), 5 controller outputs 
    N = 21 * NUM_HIDDEN + (NUM_HIDDEN + 1) * 5

    # store all fitness values of the population across generations and runs
    all_fitnesses = np.zeros((n_runs, NGEN, config.population_size))
    # store weights of HoF-best individual for each run
    best_individuals = np.zeros((n_runs, N))
    # store statistics dataframes for each run
    list_run_stats = []


    def run_evolution(run, env, config):
        """Core function running a single evolution of CMA-ES and collecting data.
        Args:
            run (int): current run number
            env (Environment): Environment object representing the the Evoman instance
            config (CMAESConfig): configuration object for CMA-ES
        Returns:
            run_fitnesses: 2D array with fitness values for each individual in each generation
            best_individual: best individual from the Hall of Fame
            run_stats: dataframe with statistics for each generation
        """
        pbar_gens = tqdm.tqdm(total=NGEN, desc=f'Run {run+1} | sigma={config.sigma:.2f}  n_pop={config.population_size:3d}', unit='gen', position=pbar_pos+run)

        # Custom seeding (NOTE: not thread-safe yet) TODO: make thread-safe with np.random.RandomState passing
        run_seed = SEED + HP_N_TRIALS + run + 1
        np.random.seed(run_seed)
        random.seed(run_seed)

        # store fitness values of population for each generation
        run_fitnesses = np.zeros((NGEN, config.population_size))

        # Reset the toolbox to not contaminate runs
        toolbox = setup_toolbox(N, env, config)

        # Initialize population and Hall of Fame
        population = toolbox.population(n=config.population_size)
        hof = tools.HallOfFame(HOF_SIZE, similar=lambda a, b: np.array_equal(a.fitness.values, b.fitness.values))

        # Initialize statistics collector
        stats = setup_data_collector()
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        # Main loop running across all generations, stopping criterion is NGEN
        for gen in range(NGEN):
            # Generate new population and evaluate fitness
            population = toolbox.generate()
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Update the strategy using population
            toolbox.update(population)
            # Update Hall of Fame
            hof.update(population)

            # Store data
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(population), **record)
            run_fitnesses[gen] = [ind.fitness.values[0] for ind in population]

            pbar_gens.update(1)

        best_individual = hof[0]
        run_stats = pd.DataFrame(logbook)
        return run_fitnesses, best_individual, run_stats


    # Run evolution for each run
    if parallel:
        raise NotImplementedError("Not yet supported. Note that hyperparameter tuning is already parallelized.")
    else:
        pbar_runs = tqdm.tqdm(total=n_runs, desc='Runs', unit='run', position=pbar_pos)
        for run in range(n_runs):
            # Run evolution once
            run_fitnesses, best_individual, run_stats = run_evolution(run, env, config)

            # Gather data
            all_fitnesses[run] = run_fitnesses
            best_individuals[run] = best_individual
            list_run_stats.append(run_stats)
            pbar_runs.update(1)

    return all_fitnesses, best_individuals, list_run_stats

def load_hyperparameters():
    """Helper function for returning hyperparameters for the training runs.
    Returns:
        config: CMAESConfig object with the hyperparameters
    """
    if HP_LOAD_FROM_FILE:
        print('Loading best hyperparameter values found during tuning from file...')
        file_path = os.path.join(folder, 'hp_best_params.csv')
        print(file_path)
        try:
            with open(file_path, 'r') as f:
                df = pd.read_csv(f)
                config = CMAESConfig(sigma=df['sigma'].values[0], population_size=df['pop_size'].values[0])
        except FileNotFoundError:
            print(f"File not found. You likely have not run tuning with {HP_FITNESS_CRITERION.__name__}.")
            sys.exit(1)
    else:
        print('Using hard-coded hyperparameter values in script...')
        config = CMAESConfig(sigma=SIGMA, population_size=POPULATION_SIZE)

    print('Hyperparameters: ', config)

    return config

def train(config, controller=CONTROLLER):
    print(f'Training the ideal controller from {N_RUNS} evolutions...')
    # Create the environment
    env = create_environment(name, enemy_set, controller)
    # Run evolution
    all_fitnesses, best_individuals, list_df_stats = run_evolutions(env, config, n_runs=N_RUNS)

    # Save data
    np.save(os.path.join(folder, 'train_all_fitnesses.npy'), all_fitnesses)
    np.save(os.path.join(folder, 'train_best_individuals.npy'), best_individuals)
    for run, stats in enumerate(list_df_stats):
        with open(os.path.join(folder, f"train_stats_run{run + 1}.csv"), 'w') as f:
            stats.to_csv(f, index=False)

    return all_fitnesses, best_individuals, list_df_stats

if __name__ == '__main__':
    # Setup
    set_names, enemy_sets = zip(*ENEMY_SETS.items())
    data_folders = [os.path.join('data', 'cmaes', f'{set_name}', f'hp_{HP_FITNESS_CRITERION.__name__}') for set_name in set_names]
    for folder in data_folders:
        os.makedirs(folder, exist_ok=True)
    
    # Core
    for name, folder, enemy_set in zip(set_names, data_folders, enemy_sets):
        config = load_hyperparameters()
        train(config)