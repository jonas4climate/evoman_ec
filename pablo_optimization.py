"""
Builds on generalist_optimization_test.py to use NEAT to train on increasingly complex topologies
"""

import random
import numpy as np
import os
import pandas as pd
import tqdm

from deap import base, creator, tools, cma
from evoman.environment import Environment
from controller_cmaes import controller_cmaes

#################################################

np.random.seed(42)

#################################################

# Folders to be created for data storage
EXP_NAME_1, EXP_NAME_2 = 'generalist_test_set_1', 'generalist_test_set_2'
DATA_FOLDER_1, DATA_FOLDER_2 = os.path.join('data', EXP_NAME_1), os.path.join('data', EXP_NAME_2)

# Two groups of enemies
ENEMY_SET_1 = [3, 5]
ENEMY_SET_2 = [2, 6] # 6, 7, 8

# Number of hidden layers in the neural network
NUM_HIDDEN = 10                                                                 # MAXIMUM NUMBER

# Number of repeated runs
N_RUNS = 2 # TODO: Change

# Number of individuals
POPULATION_SIZE = 10 # TODO: Change

# Number of generations (per topology)
NGEN = 5                                                                       # CHANGE FOR FINAL TEST

# Hall-of-fame size (number of best individuals across-all-generations saved)
HOF_SIZE = 1

# The initial standard deviation of the distribution.
SIGMA = 2.5

# Number of children to produce at each generation
LAMBDA = 10 # TODO: Change

#################################################

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

#################################################

""" We Train in segments:
- Each segment has a maximum complexity in network
- The fitness function ensures NEAT complies w/ this limit
- Then, when reaching threshold we record the performance (& assume thats optimal performance for that complexity)
- Then, update the threshold to next complexity
- Set the strategy topology at that of the previous end point
- Run agaim
- Do until we reach the complexity of the baseline


TODOs:
-   Get syntax for data storage with CMA - Yes
-   Debug data saving format so it can be visualized with the eneralist_visualization.py file
-   Substitute CMA for NEAT
"""


# Define the different complexities of the network (defined by the number of nodes in the hidden layer)
# The real number of weights of the associated network will be: Ni = 21 * n + (n + 1) * 5
# This value will be used as a mask of 0s for the weights not used.
hidden_layer_node_range = [int(i+1) for i in range(NUM_HIDDEN)]



def run_evolutions(env, n_runs=1):

    # TQDM's progress bar (including different topologies now)
    pbar_gens = tqdm.tqdm(total=n_runs*NGEN*NUM_HIDDEN, desc=f'Training generalist against enemies: {env.enemies}', unit='gen', position=1)

    # Numpy array that will store all fitness values across all runs, generations and individuals (including different topologies now)
    # Weights of topologies with fewer nodes will be set to 0
    all_fitnesses = np.zeros((n_runs, NGEN, POPULATION_SIZE,NUM_HIDDEN))

    # Numer of weights in the neural network
    # --- 21 := (no. of sensors + 1)
    # --- 5 := no. of outputs 
    N = 21 * NUM_HIDDEN + (NUM_HIDDEN + 1) * 5
    
    # TODO: ADD ZEROS TO THE NON USED WEIGHTS (SO WE CAN SAVE THEM IN SAME ARRAY!) when saving stuff

    # Numpy array that will store weights of HoF-best individual at each generation, (including different topologies now)
    best_individuals = np.zeros((n_runs, N,NUM_HIDDEN))


    for hidden_nodes in hidden_layer_node_range:

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


                #print(type(population),population)
                # EXTRA: Here we need to change the population, that has N weights, and add extra zeros padding on the middle for the unused nodes
                # This is done to not change the evoman player controller definition itself, and help save data in arrays of same size
                # I = [w0,w1...,w20,w21,...w250,b1,...,b15]
                # MEW
                population[0][20*hidden_nodes:250] = [0]                                   # First switch unused weights (first 20 for first node, second 20 for second node...)
                population[0][250:int(NUM_HIDDEN+hidden_nodes)] = [0]                      # Then switch off biases    (assumes output layer biases are last 5)


                #print(population[0],'\n')
                #print('HIIII')
                #print(1/0)


                fitnesses = list(map(toolbox.evaluate, population))
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = fit

                # Update current population and the Hall-of-Fame
                toolbox.update(population)
                hof.update(population)

                # NEW: Add AGAIN zeros manually to positions that are forced to be 0
                population[0][20*hidden_nodes:250] = [0]                                   # First switch unused weights (first 20 for first node, second 20 for second node...)
                population[0][250:int(NUM_HIDDEN+hidden_nodes)] = [0]                      # Then switch off biases    (assumes output layer biases are last 5)




                # Save the data in Deap's Statistics object
                record = stats.compile(population)

                # Create logs
                logbook.record(gen=gen, nevals=len(population), **record)

                # Save fitnesses from the current generation
                all_fitnesses[run, gen,:,int(hidden_nodes-1)] = [ind.fitness.values[0] for ind in population]

                # Update the progress bar
                pbar_gens.update(1)

            # Save the best individual from the current run
            best_individuals[run,:,int(hidden_nodes-1)] = hof[0]                           # TODO: Fix saving format so it can be more easily opened later (ex. split into one file per topology in main)
        
    # Create the stats dataframe based on Deap's logbook
    df_stats = pd.DataFrame(logbook)

    # Close the progress bar
    pbar_gens.close()

    return all_fitnesses, best_individuals, df_stats







#################################################

if __name__ == '__main__':

    # Create environments
    ENV_1 = create_environment(EXP_NAME_1, ENEMY_SET_1)
    ENV_2 = create_environment(EXP_NAME_2, ENEMY_SET_2)

    # Folder setup
    os.makedirs(DATA_FOLDER_1, exist_ok=True)
    os.makedirs(DATA_FOLDER_2, exist_ok=True)

    # Deap setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax) # type: ignore
    
    for (env, data_folder) in zip([ENV_1, ENV_2], [DATA_FOLDER_1, DATA_FOLDER_2]):

        # Run the training loop
        all_fitnesses, best_individuals, df_stats = run_evolutions(env, N_RUNS)

        # TODO: Save the data from each TOPOLOGY too
        # Save the data for each run
        for run in range(1, N_RUNS + 1):
            # Create folder if doesn't exist yet
            os.makedirs(data_folder, exist_ok=True)

            # Save all fitness across all runs
            np.save(os.path.join(data_folder, f'all_fitnesses.npy'), all_fitnesses)

            # Save best individuals
            np.save(os.path.join(data_folder, f'best_individual_run{run}.npy'), best_individuals[run - 1])

            # Save statistics
            df_stats.to_csv(os.path.join(data_folder, f'stats_run{run}.csv'), index=False)
