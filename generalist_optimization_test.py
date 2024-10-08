import random
import numpy as np
import os
import pandas as pd

from tqdm.auto import tqdm
from deap import base, creator, tools, cma
from evoman.environment import Environment
from controller_cmaes import controller_cmaes

# Folders to be created for data storage
EXP_NAME_1, EXP_NAME_2 = 'generalist_test_set_1', 'generalist_test_set_2'
DATA_FOLDER_1, DATA_FOLDER_2 = os.path.join('data', EXP_NAME_1), os.path.join('data', EXP_NAME_2)

# Two groups of enemies
ENEMY_SET_1 = [3, 5, 7]
ENEMY_SET_2 = [2, 6, 7, 8]

# Number of hidden layers in the neural network
NUM_HIDDEN = 10

# Number of repeated runs
N_RUNS = 5 # TODO: Change

# Number of individuals
POPULATION_SIZE = 100 # TODO: Change

# Number of generations
NGEN = 100 # TODO: Change

# Hall-of-fame size (number of best individuals across-all-generations saved)
HOF_SIZE = 1

# The initial standard deviation of the distribution.
SIGMA = 2.5

# Number of children to produce at each generation
LAMBDA = 100 # TODO: Change

np.random.seed(42)

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
    agg_fitness, _, _, _ = environment.play()
    return agg_fitness,

def create_environment(experiment_name, enemy_set, visuals=False):
    """Returns an Environment object for the Evoman framework.

    Args:
        experiment_name (string): name of the experiment, used for logging
        enemy_set (List[int]): list with enemies to train on
        visuals (bool, optional): Whether to display environment to the screen. Defaults to False.
    """
    return Environment(experiment_name=experiment_name,
                        enemies=enemy_set,
                        multiplemode="yes",
                        enemymode='static',
                        speed='normal' if visuals else 'fastest',
                        player_controller=controller_cmaes(),
                        savelogs="no",
                        clockprec="low",
                        visuals=visuals)

def setup_toolbox(N):
    """Helper function for setting up toolbox (setup) from the Deap package.

    Args:
        N (int): no. of weights required for the neural network
    """
    # Create a new toolbox (setup)
    toolbox = base.Toolbox()

    # Creates the initial population of individuals
    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fitness, environment=env)

    # CMA-ES setup
    centroid = np.zeros(N)
    strategy = cma.Strategy(centroid=centroid, sigma=SIGMA, lambda_=LAMBDA)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    return toolbox

def setup_data_collector():
    """Helper function for setting up data collector from the Deap package.

    Args:
        stats (deap.tools.Statistics): 
    """
    # Creates Statistics object and binds it to fitness values
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    # Sets up data to be registered at each step: mean, std and min / max fitness in population
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    return stats

def run_evolutions(env, n_runs=1):

    # TQDM's progress bar
    pbar_gens = tqdm(total=n_runs*NGEN, desc=f'Training generalist against enemies: {env.enemies}', unit='gen')

    # Numer of weights in the neural network
    # --- 21 := (no. of sensors + 1)
    # --- 5 := no. of outputs 
    N = 21 * NUM_HIDDEN + (NUM_HIDDEN + 1) * 5

    # Numpy array that will store all fitness values across all runs, generations and individuals
    all_fitnesses = np.zeros((n_runs, NGEN, POPULATION_SIZE))

    # Numpy array that will store weights of HoF-best individual at each generation
    best_individuals = np.zeros((n_runs, N))

    for run in range(n_runs):

        # Reset the toolbox
        toolbox = setup_toolbox(N)
        
        # Create initial population
        population = toolbox.population(n=POPULATION_SIZE)

        # Setup the Hall-of-Fame
        hof = tools.HallOfFame(HOF_SIZE, similar=lambda a, b: np.array_equal(a.fitness.values, b.fitness.values))

        # Setup the data collector
        stats = setup_data_collector()

        # Logging
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        # Main loop running across all generations
        for gen in range(NGEN):

            # Generate new individuals and evaluate their fitness
            population = toolbox.generate()
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Update current population and the Hall-of-Fame
            toolbox.update(population)
            hof.update(population)

            # Save the data in Deap's Statistics object
            record = stats.compile(population)

            # Create logs
            logbook.record(gen=gen, nevals=len(population), **record)

            # Save fitnesses from the current generation
            all_fitnesses[run, gen] = [ind.fitness.values[0] for ind in population]

            # Update the progress bar
            pbar_gens.update()

        # Save the best individual from the current run
        best_individuals[run] = hof[0]
    
    # Create the stats dataframe based on Deap's logbook
    df_stats = pd.DataFrame(logbook)

    # Close the progress bar
    pbar_gens.close()

    return all_fitnesses, best_individuals, df_stats

if __name__ == '__main__':

    # Create environments
    ENV_1 = create_environment(EXP_NAME_1, ENEMY_SET_1)
    ENV_2 = create_environment(EXP_NAME_2, ENEMY_SET_2)

    # Deap setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax) # type: ignore
    
    for (env, data_folder) in zip([ENV_1, ENV_2], [DATA_FOLDER_1, DATA_FOLDER_2]):

        # Run the training loop
        all_fitnesses, best_individuals, df_stats = run_evolutions(env, N_RUNS)

        # Delete folder and create folder to reset its contents
        for file in os.listdir(data_folder):
            os.remove(os.path.join(data_folder, file))
        os.removedirs(data_folder)
        os.makedirs(data_folder)

        # Save the data for each run
        for run in range(1, N_RUNS + 1):

            # Save all fitness across all runs
            np.save(os.path.join(data_folder, f'all_fitnesses.npy'), all_fitnesses)

            # Save best individuals
            np.save(os.path.join(data_folder, f'best_individual_run{run}.npy'), best_individuals[run - 1])

            # Save statistics
            df_stats.to_csv(os.path.join(data_folder, f'stats_run{run}.csv'), index=False)
