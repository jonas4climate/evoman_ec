import random
import numpy as np
import os
import sys
import pandas as pd
import tqdm

import optuna
from optuna.samplers import TPESampler
from deap import base, creator, tools, algorithms, cma

from evoman.environment import Environment
from controller_cmaes import controller_cmaes

# Folders
EXP_NAME = 'cmaes'
DATA_FOLDER = os.path.join('data', EXP_NAME)

# Experiment parameters
ENEMIES = [1, 3, 4]  # TODO: make larger pick 3
ENEMY_MODE = 'static'

# Custom fitness parameters
FITNESS_GAMMA = 0.75
FITNESS_ALPHA = 0.25

# Controller parameters
NUM_HIDDEN = 10
NUM_OUTPUTS = 5

# Evolutionary algorithm parameters
POPULATION_SIZE = 100
LAMBDA = 100
HOF_SIZE = 1
NGEN = 100
SIGMA = 2.5

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
    pbar_gens = tqdm.tqdm(total=n_runs*NGEN, desc=f'Training specialist against enemy {env.enemies[0]}', unit='gen', position=1)
    all_fitnesses = np.zeros((n_runs, NGEN, POPULATION_SIZE))
    # decision format stored: [left, right, jump, shoot, release, left_or_right, jump_or_release]
    all_decisions = np.zeros((n_runs, NGEN, 2, 7)) # mean and std for each type of decision at each generation & run
    for run in range(n_runs):
        toolbox = base.Toolbox()  # Reset the toolbox to not contaminate runs

        num_inputs = env.get_num_sensors()
        n = (num_inputs+1) * NUM_HIDDEN + (NUM_HIDDEN+1) * NUM_OUTPUTS

        toolbox.register("attr_float", random.uniform, -1.0, 1.0)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n) # type: ignore
        toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore
        toolbox.register("evaluate", eval_fitness, env=env)

        # CMA-ES
        centroid = np.zeros(n)
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
            decision_history = np.array(env.player_controller.decision_history)
            dec_mean, dec_std = np.mean(decision_history, axis=0), np.std(decision_history, axis=0)
            all_decisions[run, gen] = [dec_mean, dec_std]
            env.player_controller.reset_history()

            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(population), **record)
            all_fitnesses[run, gen] = [ind.fitness.values[0] for ind in population]
            # print(logbook.stream) # In the way of tqdm
            pbar_gens.update(1)

        # Save data
        game_folder = os.path.join(DATA_FOLDER, str(env.enemies[0]))
        os.makedirs(game_folder, exist_ok=True)
        np.save(os.path.join(game_folder, f'best_individual_run{run}_{ENEMY_MODE}.npy'), hof[0])
        df_stats = pd.DataFrame(logbook)
        df_stats.to_csv(os.path.join(game_folder, f'stats_run{run}_{ENEMY_MODE}.csv'), index=False)
    
    np.save(os.path.join(game_folder, f'all_decisions_{ENEMY_MODE}.npy'), all_decisions)
    np.save(os.path.join(game_folder, f'all_fitnesses_{ENEMY_MODE}.npy'), all_fitnesses)
    pbar_gens.close()
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
                            player_controller=controller_cmaes(log_history=True),
                            visuals=False) for enemy in ENEMIES]
    
    # Number of runs for training, testing and tuning
    n_runs = 1
    if '--runs' in sys.argv:
        n_runs = int(sys.argv[sys.argv.index('--runs') + 1])

    test_n_repeats = 5
    if '--repeats' in sys.argv:
        test_n_repeats = int(sys.argv[sys.argv.index('--repeats') + 1])

    tuning_n_trials = 10
    if '--ntrials' in sys.argv:
        tuning_n_trials = int(sys.argv[sys.argv.index('--ntrials') + 1])

    if '--tune' in sys.argv:
        def hyerparam_trial_run(trial):
            # "Sample" hyperparameters
            global LAMBDA, POPULATION_SIZE, N_GEN, SIGMA
            SIGMA = trial.suggest_float('sigma', 0.1, 10.0)
            LAMBDA = POPULATION_SIZE = trial.suggest_int('pop_size', 10, 1000)
            N_GEN = trial.suggest_int('n_gen', 10, 1000)

            # Run evolutions
            # TODO: needs adjustment for specialists
            for env in envs:
                run_evolutions(env, n_runs)

            # TODO: Determine fitness of hyperparamaters e.g. by average fitness and/or 
            # max fitness obtained across environments and runs or perhaps using multi-objective optimization?
            # TODO: Needs restructuring for group training
            all_fitnesses = np.load(os.path.join(DATA_FOLDER, f'{envs[0].enemies[0]}/all_fitnesses_{ENEMY_MODE}.npy'))
            fitness = np.mean(all_fitnesses[:, -1, -1]) # mean of last run
            return fitness

        # Select bayesian optimization algorithm (default is Tree-structured Parzen Estimator)
        sampler = TPESampler(seed=SEED)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(hyerparam_trial_run, n_trials=tuning_n_trials)
        
        df = study.trials_dataframe()
        df.to_csv(os.path.join(DATA_FOLDER, 'tuning_results.csv'), index=False)

        best_params = study.best_params
        SIGMA = best_params['sigma']
        LAMBDA = POPULATION_SIZE = best_params['pop_size']
        N_GEN = best_params['n_gen']
        df = pd.DataFrame(best_params, index=[0])
        df.to_csv(os.path.join(DATA_FOLDER, 'best_params.csv'), index=False)

    if '--train' in sys.argv:
        pbar_games = tqdm.tqdm(total=len(envs), desc='Training on games', unit='game', position=0)
        for env in envs:
            run_evolutions(env, n_runs)
            pbar_games.update(1)

    if '--test' in sys.argv:
        for (i, env) in tqdm.tqdm(enumerate(envs), total=len(envs), desc='Testing on games', unit='game', position=0):
            gains = np.zeros(test_n_repeats * n_runs)

            for j in range(n_runs):
                with open(os.path.join(DATA_FOLDER, f"{env.enemies[0]}/best_individual_run{j}_static.npy"), 'rb') as f:
                    optimal_weights = np.load(f)
                env.player_controller.set_weights(optimal_weights, NUM_HIDDEN)

                for k in range(test_n_repeats):
                    _, pl, el, _ = env.play()
                    # Gain = player life - enemy life
                    gains[j*test_n_repeats + k] = pl - el

            # Save the gains
            np.save(os.path.join(DATA_FOLDER, f'{env.enemies[0]}', 'gains.npy'), gains)

    if '--watch' in sys.argv:
        for env in envs:
            for run in range(n_runs):
                file = os.path.join(DATA_FOLDER, str(env.enemies[0]), f'best_individual_run{run}_{ENEMY_MODE}.npy')
                best_individual = np.load(file, allow_pickle=True)
                env.player_controller.set_weights(best_individual, NUM_HIDDEN) # type: ignore
                print(f'Game {env.enemies} started, go watch!')
                env.visuals = True
                env.speed = "normal"
                env.play()