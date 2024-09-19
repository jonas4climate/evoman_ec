###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# Standard libraries
import sys
import os
import time
import numpy as np

# Evoman framework
### Game environment
from evoman.environment import Environment
### Player's neural network structure
from tutorial.demo_controller import player_controller

# Constants
EXPERIMENT_NAME = "optization_bg"
N_HIDDEN_NEURONS = 10

# Create experiment directory in case it's missing
if not os.path.exists(EXPERIMENT_NAME):
    os.makedirs(EXPERIMENT_NAME)

# Number of total weights (parameters) in Player controller (neural network)
n_vars = 21 * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5

# Genetic algorithm parameters
population_size = 20
# n_evaluations = 3 ???
# last_best = 0 ???
n_offspring = 10
n_generations = 40
weight_upper_bound = 1
weight_lower_bound = -1
mutation_sigma = 0.4

# Initializes the population ensuring it's in bounds
def initialize_population(population_size, lower, upper, n_weights):
    return np.random.uniform(lower, upper, (population_size, n_weights))

# Runs simulation, returns the fitness
def simulation(env, player_controller):
    f, _, _, _ = env.play(pcont=player_controller)
    return f

# Parent selection: chance of becoming a parent correlated w/ fitness
def parent_selection(population, pop_fit, n_parents, smoothing = 1):
    fitness = normalize(pop_fit)
    fitness = fitness + smoothing - np.min(fitness)

    # Fitness proportional selection probability
    fps = fitness / np.sum(fitness)

    # Make a random selection of indices
    parent_indices = np.random.choice (np.arange(0, population.shape[0]), (n_parents, 2), p=fps)
    return population[parent_indices]

# Normalizes fitness values
def normalize(pop_fit):
    vmin = np.min(pop_fit)
    vmax = np.max(pop_fit)

    if (vmax - vmin) > 0:
        return (pop_fit - vmin) / (vmax - vmin)
    else:
        return pop_fit / pop_fit


# Evaluation: Endgame fitness of each individual in the population
def evaluate(env, population):
    return np.array(list(map(lambda pcont: simulation(env, pcont), population)))

# Tournament selection
def tournament(population, pop_fit):
    c1 =  np.random.randint(0, population.shape[0], 1)
    c2 =  np.random.randint(0, population.shape[0], 1)

    if pop_fit[c1] > pop_fit[c2]:
        return population[c1][0]
    else:
        return population[c2][0]

# Keeps weights in bounds during mutation
def limits(weight):
    if weight > weight_upper_bound:
        return weight_upper_bound
    elif weight < weight_lower_bound:
        return weight_lower_bound
    else:
        return weight
    
# Mutation operation: adds small Gaussian noise to each gene, while keeping them in bounds
def mutate(pop, min_value, max_value, sigma):
    mutation = np.random.normal(0, sigma, size=pop.shape)
    new_pop = pop + mutation
    new_pop[new_pop > max_value] = max_value
    new_pop[new_pop < min_value] = min_value
    return new_pop

# Crossover operation: with equal chance take each gene from one of two parents
def crossover(parents):
    parentsA, parentsB = np.hsplit(parents, 2)
    roll = np.random.uniform(size = parentsA.shape)
    offspring = parentsA * (roll >= 0.5) + parentsB * (roll < 0.5)
    # Squeeze to get rid of the extra dimension created during parent selecting
    return np.squeeze(offspring, 1)

# Leaves out the top npop individuals in the population
def survivor_selection(pop, pop_fit, n_pop):
    best_fit_indices = np.argsort(pop_fit * -1) # -1 since we are maximizing
    survivor_indices = best_fit_indices [:n_pop]
    return pop[survivor_indices], pop_fit[survivor_indices]

# Initializes the environment with the given enemy type and their difficulty
def initialize_environment(enemy_id, difficulty):
    return Environment(experiment_name=EXPERIMENT_NAME,
                  enemies=[enemy_id],
                  playermode="ai",
                  player_controller=player_controller(N_HIDDEN_NEURONS),
                  enemymode="static",
                  level=difficulty,
                  speed="fastest",
                  visuals=False)

# Trains the neural network for the given environment (allows for variable enemy ID and difficulty).
# Returns the optimal weight vector.
def train(env):
    pop = initialize_population(population_size, weight_lower_bound, weight_upper_bound, n_vars)
    pop_fit = evaluate(env, pop)

    for gen in range (n_generations):
        parents = parent_selection(pop, pop_fit, n_offspring)
        offspring = crossover(parents)
        offspring = mutate(offspring, weight_lower_bound, weight_upper_bound, mutation_sigma)

        offspring_fit = evaluate(env, offspring)

        # Concatenating to form a new population
        pop = np.vstack((pop, offspring))
        pop_fit = np.concatenate([pop_fit, offspring_fit])

        pop, pop_fit = survivor_selection(pop, pop_fit, population_size)
        print (f"Gen {gen} - Best: {np.max (pop_fit)} - Mean: {np.mean(pop_fit)}")

    print()
    return pop[np.argmax(pop_fit)]

ENEMIES = range(1,9)
DIFFICULTY = 2

for en in ENEMIES:
    print(f"#### Simulation: ENEMY ID = {en}")

    env = initialize_environment(en, DIFFICULTY)
    optimal_weights = train(env)

    np.save(EXPERIMENT_NAME + f"/ENEMY_{en}_DIFF_{DIFFICULTY}", optimal_weights)