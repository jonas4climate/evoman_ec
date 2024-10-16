"""
TODO:
- make something better than the checkpoints list currently using
- make sure that the loading of population is working
- try out the --test option ?
- hyperparameter optimization 
    - I think mainly node_add_prob is interesting + parameters that are similar to what is used in CMA?
- keeping track of time per generation (or at least extract that data) + improvement of fitness for "computational efficiency"
"""

import os
import sys
import neat
import pandas as pd
import numpy as np
import pickle
import tqdm

from time import time as timefunction # NEW TIME TRACKING (See https://realpython.com/python-timer/)

from generalist_shared import create_environment

# Load configuration
from generalist_neat_config import *

np.random.seed(SEED)

def evaluate_individual(id, network, env):
    agg_fit, p_life, e_life, time = env.play(pcont=network)
    return agg_fit

def eval_genomes(genomes, config, run, stats_data, fitnesses_data, n_nodes_data, n_weights_data, generation_times_data, pbar_gens, generation, env):  
    """Evaluate the fitness of a list of genomes. Called as part of the NEAT evolution process.
    
    Args:
        genomes (list): list of genomes to evaluate
        config (neat.Config): configuration object for NEAT
        run (int): current run number
        stats_data (list): list to store statistics data
        fitnesses_data (np.array): array to store fitness data
        n_nodes_data (np.array): array to store number of nodes data
        n_weights_data (np.array): array to store number of weights data
        pbar_gens (tqdm.tqdm): progress bar for generations
        generation (int): current generation number
        env (Environment): Evoman environment
    """      
    n = len(genomes)
    fitnesses = np.zeros(n)
    n_nodes = np.zeros(n)
    n_weights = np.zeros(n)
    genome_times = np.zeros(n)                                                                       # NEW: TIME TRACKING, define array for genome times
    best_network = None
    max_fitness = - np.inf

    for i, (id, genome) in enumerate(genomes):

        time_start = timefunction()                                                              # NEW TIME TRACKING, start genome counter
            
        # create a NN based on the genome provided
        net = neat.nn.FeedForwardNetwork.create(genome, config)    
        num_nodes = len(net.node_evals)
        fitness = evaluate_individual(id, net, env) # Calculate fitness based on network's success
        genome.fitness = fitness  # Assign fitness to genome
        # keep track of highest fitness and best genome 
        if fitness > max_fitness:
            max_fitness = fitness
            best_network = net
            best_id = id

        fitnesses[i] = genome.fitness # To keep track of fitness
        n_weights[i] = len(genome.connections)
        n_nodes[i] = num_nodes

        time_total = timefunction()-time_start                                           # NEW TIME TRACKING, end timer of current genome
        genome_times[i] = time_total                                               # NEW TIME TRACKING, save current genome time

    generation_times_data[run, generation, :len(genome_times)] = genome_times
    n_nodes_data[run, generation, :len(n_nodes)] = [np.mean(n_nodes[i]) for i in range(len(n_nodes))]
    n_weights_data[run, generation, :len(n_weights)] = [np.mean(n_weights[i]) for i in range(len(n_weights))]
    fitnesses_data[run, generation, :len(fitnesses)] = fitnesses
    
    mean_number_nodes = np.mean(n_nodes)
    max_fitness = max(fitnesses)
    mean_fitness = np.mean(fitnesses)
    std_fitness = np.std(fitnesses)
    
    row_data = [generation, max_fitness, mean_fitness, std_fitness]
    stats_data.append(row_data)

    # print(f"In this generation the best network has {len(best_network.node_evals)} nodes. ID {best_id}. Fitness {max_fitness}")
    pbar_gens.update(1)

def run_evolutions(env, config, name, pbar_pos=2, n_runs=N_RUNS, n_gens=NGEN, show_output=True):
    """Run multiple evolutions using NEAT for the given environment to generate a controlelr and returns data and solutions.
    
    Args:
        env (Environment): Evoman environment
        name (str): name of the environment
        n_runs (int): number of runs to perform
    Returns:
        all_fitnesses (np.array): fitnesses of all individuals in all generations
    """                                                                # NEW TIME TRACKING (notice popsize is NOT doubled!)
    pop_size = int(config.pop_size * 2.0) # 100% margin for growth
    all_fitnesses = np.full((n_runs, n_gens, pop_size), np.nan)
    n_nodes_data = np.full((n_runs, n_gens, pop_size), np.nan)
    n_weights_data = np.full((n_runs, n_gens, pop_size), np.nan)
    generation_times_data = np.full((n_runs, n_gens, pop_size), np.nan)  
    best_individuals = []
    list_df_stats = []
    
    pbar_gens = tqdm.tqdm(total=n_runs*n_gens, desc=f'(Process {pbar_pos-1}) Training generalist against enemies {name}', unit='gen', position=pbar_pos)
        
    # COMPLEXITY_INDEX = 1

    for run in range(n_runs):
        stats_data = []

        # Create population (top-level object for a NEAT run) and fitness function
        p = neat.Population(config)

        # Statistic gathering and output
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        if show_output:
            p.add_reporter(neat.StdOutReporter(True))

        # Fitness function
        f_fitness = lambda genomes, config: eval_genomes(genomes, config, run, stats_data, all_fitnesses, n_nodes_data, n_weights_data, generation_times_data, pbar_gens, p.generation, env)

        # Run evolution
        best_individual = p.run(f_fitness, n_gens)

        # Data handling
        best_individuals.append(best_individual)
        df_stats = pd.DataFrame(stats_data)
        df_stats.columns = ['generation', 'max', 'mean', 'std']
        list_df_stats.append(df_stats)
        # TODO Append new 100 elements
        
    return all_fitnesses, best_individuals, list_df_stats, n_nodes_data, n_weights_data, generation_times_data

def get_config(path=CONFIG_PATH):
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            path)

def train(name, folder, enemy_set, save_data=True, controller=CONTROLLER):
    env = create_environment(name, enemy_set, controller)
    config = get_config()
    all_fitnesses, best_individuals, list_df_stats, n_nodes_data, n_weights_data, generation_times_data = run_evolutions(env, config, name)

    # Save data
    if save_data:
        np.save(os.path.join(folder, f'all_fitnesses_{ENEMY_MODE}.npy'), all_fitnesses)
        np.save(os.path.join(folder, f'all_n_nodes_{ENEMY_MODE}.npy'), n_nodes_data)
        np.save(os.path.join(folder, f'all_n_weights_{ENEMY_MODE}.npy'), n_weights_data)
        # np.save(os.path.join(folder, f'all_n_times_{ENEMY_MODE}.npy'), generation_times_data)           # NEW TIME TRACKING
        
        for run, (best_individual, df_stats) in enumerate(zip(best_individuals, list_df_stats)):
            with open(os.path.join(folder, f'best_individual_run{run}_{ENEMY_MODE}.pkl'), 'wb') as f:
                pickle.dump(best_individual, f)
            df_stats.to_csv(os.path.join(folder, f'stats_run{run}_{ENEMY_MODE}.csv'), index=False)

    return all_fitnesses, best_individuals, list_df_stats

if __name__ == '__main__':
    # Setup
    set_names, enemy_sets = zip(*ENEMY_SETS.items())
    data_folders = [os.path.join('data', 'neat', f'{set_name}') for set_name in set_names]
    for folder in data_folders:
        os.makedirs(folder, exist_ok=True)

    for name, folder, enemy_set in zip(set_names, data_folders, enemy_sets):
        game_folder = os.path.join(DATA_FOLDER, str(name))
        os.makedirs(game_folder, exist_ok=True)

        train(name, folder, enemy_set)