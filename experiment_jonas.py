import random
import numpy as np
import os
import sys
from deap import base, creator, tools, algorithms, cma
from evoman.environment import Environment
from experiment_util import *
from controller_jonas import controller_jonas

# Experiment parameters
EXP_NAME = 'data/experiment_jonas'
ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]  # pick 3
ENEMY_MODE = 'static'

# Controller parameters
NUM_HIDDEN = 20
NUM_OUTPUTS = 5

# Evolutionary algorithm parameters
POPULATION_SIZE = 100
LAMBDA = 100
HOF_SIZE = 5
NGEN = 150
SIGMA = 10
TOURNAMENT_SIZE = 5
MUTATE_MU = 0
MUTATE_SIGMA = 0.1
MUTATE_INDPB = 0.2
MATE_INDPB = 0.2

os.makedirs(EXP_NAME, exist_ok=True)

def evaluate_fitness(individual, env):
    env.player_controller.set_weights(individual, NUM_HIDDEN)
    fitness, player_life, enemy_life, game_run_time = env.play()
    return fitness,

def evaluate_fitness_custom(individual, env):
    env.player_controller.set_weights(individual, NUM_HIDDEN)
    _, player_life, enemy_life, game_run_time = env.play()
    custom_fitness = 0.75 * (100 - enemy_life) + 0.25 * player_life - 2*np.log(game_run_time)
    return custom_fitness,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

def similar(ind1, ind2):
    return np.array_equal(ind1.fitness.values, ind2.fitness.values)

def run_evolution(env):
    toolbox = base.Toolbox()  # Reset the toolbox to not contaminate runs

    num_inputs = env.get_num_sensors()
    n = (num_inputs+1) * NUM_HIDDEN + (NUM_HIDDEN+1) * NUM_OUTPUTS

    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_fitness_custom, env=env)
    toolbox.register("mate", tools.cxUniform, indpb=MATE_INDPB)
    toolbox.register("mutate", tools.mutGaussian, mu=MUTATE_MU, sigma=MUTATE_SIGMA, indpb=MUTATE_INDPB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # CMA-ES
    centroid = np.zeros(n)
    # centroid = np.random.randn(n) * 0.1
    strategy = cma.Strategy(centroid=centroid, sigma=SIGMA, lambda_=LAMBDA)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    population = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(HOF_SIZE, similar=similar)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("sigma", lambda _: strategy.sigma)

    # Evolutionary base loop
    final_pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof, verbose=True)

    game_folder = os.path.join(EXP_NAME, str(env.enemies[0]))
    os.makedirs(game_folder, exist_ok=True)

    np.save(os.path.join(game_folder, f'best_individual_{ENEMY_MODE}.npy'), hof[0])
    df = pd.DataFrame(logbook)
    df.to_csv(os.path.join(game_folder, f'stats_{ENEMY_MODE}.csv'), index=False)

    return population, stats, hof

if __name__ == '__main__':
    envs = [Environment(experiment_name=EXP_NAME,
                            enemies=[enemy],
                            multiplemode="no",
                            enemymode=ENEMY_MODE,
                            randomini="no",
                            savelogs="no",
                            timeexpire=3000,
                            clockprec="low",
                            player_controller=controller_jonas(),
                            visuals=False) for enemy in ENEMIES]
    save_path = os.path.join(EXP_NAME, 'best_individuals.npy')

    if '--train' in sys.argv:
        best_individuals = []
        for env in envs:
            population, stats, hof = run_evolution(env)
            best_individuals.append(hof[0])

    if '--watch' in sys.argv:
        for env in envs:
            file = os.path.join(EXP_NAME, str(env.enemies[0]), f'best_individual_{ENEMY_MODE}.npy')
            best_individual = np.load(file, allow_pickle=True)
            env.player_controller.set_weights(best_individual, NUM_HIDDEN)
            print(f'Game {env.enemies} started, go watch!')
            watch_controller_play([env], n_games=1, speed="normal")