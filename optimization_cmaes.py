import random
import numpy as np
import os
import sys
import pandas as pd
import tqdm

from deap import base, creator, tools, algorithms, cma
from evoman.environment import Environment
from experiment_util import *
from controller_jonas import controller_jonas

# Folders
EXP_NAME = 'cmaes'
DATA_FOLDER = os.path.join('data', EXP_NAME)

# Experiment parameters
ENEMIES = [1,3,4]  # TODO: make larger pick 3
ENEMY_MODE = 'static'

# Custom fitness parameters
FITNESS_GAMMA = 0.75
FITNESS_ALPHA = 0.25

# Controller parameters
NUM_HIDDEN = 10
NUM_OUTPUTS = 5

# Evolutionary algorithm parameters
POPULATION_SIZE = 50
LAMBDA = 50
HOF_SIZE = 5
NGEN = 100
SIGMA = 10
TOURNAMENT_SIZE = 5
MUTATE_MU = 0
MUTATE_SIGMA = 0.1
MUTATE_INDPB = 0.2
MATE_INDPB = 0.2

SEED = 42
np.random.seed(SEED)


def similar(ind1, ind2):
    return np.array_equal(ind1.fitness.values, ind2.fitness.values)

def evaluate_fitness(individual, env, gamma=0.9, alpha=0.1):
    env.player_controller.set_weights(individual, NUM_HIDDEN)
    _, player_life, enemy_life, game_run_time = env.play()
    custom_fitness = gamma * (100 - enemy_life) + alpha * player_life - np.log(game_run_time)
    return custom_fitness,

# Setup
os.makedirs(DATA_FOLDER, exist_ok=True)
eval_fitness = lambda ind, env: evaluate_fitness(ind, env, gamma=FITNESS_GAMMA, alpha=FITNESS_ALPHA)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax) # type: ignore


def run_evolutions(env, n_runs=1):
    pbar = tqdm.tqdm(total=n_runs*NGEN, desc=f'Training specialist against enemy {env.enemies[0]}', unit='gen', position=1)
    all_fitnesses = np.zeros((n_runs, NGEN, POPULATION_SIZE))
    for run in range(n_runs):
        toolbox = base.Toolbox()  # Reset the toolbox to not contaminate runs

        num_inputs = env.get_num_sensors()
        n = (num_inputs+1) * NUM_HIDDEN + (NUM_HIDDEN+1) * NUM_OUTPUTS

        toolbox.register("attr_float", random.uniform, -1.0, 1.0)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n) # type: ignore
        toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore

        toolbox.register("evaluate", eval_fitness, env=env)
        toolbox.register("mate", tools.cxUniform, indpb=MATE_INDPB)
        toolbox.register("mutate", tools.mutGaussian, mu=MUTATE_MU, sigma=MUTATE_SIGMA, indpb=MUTATE_INDPB)
        toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

        # CMA-ES
        centroid = np.zeros(n)
        # centroid = np.random.randn(n) * 0.1
        strategy = cma.Strategy(centroid=centroid, sigma=SIGMA, lambda_=LAMBDA)
        toolbox.register("generate", strategy.generate, creator.Individual) # type: ignore
        toolbox.register("update", strategy.update)

        population = toolbox.population(n=POPULATION_SIZE) # type: ignore
        hof = tools.HallOfFame(HOF_SIZE, similar=similar)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("sigma", lambda _: strategy.sigma)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        # Hidden base loop
        # final_pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof, verbose=True)
        
        # Exposed loop
        for gen in range(NGEN):
            population = toolbox.generate()
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            toolbox.update(population)
            hof.update(population)

            # Data collection
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(population), **record)
            all_fitnesses[run, gen] = [ind.fitness.values[0] for ind in population]
            # print(logbook.stream) # In the way of tqdm
            pbar.update(1)

        # Save data
        game_folder = os.path.join(DATA_FOLDER, str(env.enemies[0]))
        os.makedirs(game_folder, exist_ok=True)
        np.save(os.path.join(game_folder, f'best_individual_run{run}_{ENEMY_MODE}.npy'), hof[0])
        df_stats = pd.DataFrame(logbook)
        df_stats.to_csv(os.path.join(game_folder, f'stats_run{run}_{ENEMY_MODE}.csv'), index=False)
    
    np.save(os.path.join(game_folder, f'all_fitnesses_{ENEMY_MODE}.npy'), all_fitnesses)
    pbar.close()
    return

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

    if '--train' in sys.argv:
        runs = 1
        if '--runs' in sys.argv:
            runs = int(sys.argv[sys.argv.index('--runs') + 1])
        pbar = tqdm.tqdm(total=len(envs), desc='Training on games', unit='game', position=0)
        for env in envs:
            run_evolutions(env, runs)
            pbar.update(1)

    if '--watch' in sys.argv:
        for env in envs:
            file = os.path.join(DATA_FOLDER, str(env.enemies[0]), f'best_individual_{ENEMY_MODE}.npy')
            best_individual = np.load(file, allow_pickle=True)
            env.player_controller.set_weights(best_individual, NUM_HIDDEN) # type: ignore
            print(f'Game {env.enemies} started, go watch!')
            env.visuals = True
            env.speed = "normal"
            env.play()