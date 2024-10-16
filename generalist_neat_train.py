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

from evoman.environment import Environment
from controller_neat import controller_neat
from generalist_shared import create_environment

EXP_NAME = 'neat_generalist'
DATA_FOLDER = os.path.join('data', EXP_NAME)
ENEMY_MODE = 'static'
CONFIG_PATH = os.path.join('neat-config-feedforward.ini')


# Two groups of enemies
ENEMY_SETS = {
    'set_1': [3, 5, 7],
    'set_2': [2, 6, 7, 8]
}

# Custom fitness parameters
FITNESS_GAMMA = 0.75
FITNESS_ALPHA = 0.25

GEN_INTERVAL_LOG = 10
NGEN = 100 # TODO: make larger in practice

SEED = 42
np.random.seed(SEED)

os.makedirs(DATA_FOLDER, exist_ok=True)

def eval_genomes(genomes, config, run, stats_data, fitnesses_data, n_nodes_data, n_weights_data, pbar_gens, generation):        
    fitnesses = []
    n_nodes = []
    n_weights = []
    best_network = None
    max_fitness = - np.inf
    for id, genome in genomes:    
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

        fitnesses.append(genome.fitness) # To keep track of fitness
        n_weights.append(len(genome.connections))
        n_nodes.append(num_nodes)


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

def evaluate_individual(id, network, env):
    agg_fit, p_life, e_life, time = env.play(pcont=network)
    return agg_fit

def run_evolutions(n_runs, name):
    # Find out pop size
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            CONFIG_PATH)
    pop_size = int(config.pop_size * 2.0) # 100% margin for growth
    fitnesses_data = np.full((n_runs, NGEN, pop_size), np.nan)
    n_nodes_data = np.full((n_runs, NGEN, pop_size), np.nan)
    n_weights_data = np.full((n_runs, NGEN, pop_size), np.nan)

    game_folder = os.path.join(DATA_FOLDER, str(name))
    pbar_gens = tqdm.tqdm(total=n_runs*NGEN, desc=f'Training generalist against enemies {name}', unit='gen', position=1)

        
    COMPLEXITY_INDEX = 1

    for run in range(n_runs):
        stats_data = []

        # Load configuration.
        
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                CONFIG_PATH)
            
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.StdOutReporter(True))
        f_fitness = lambda genomes, config: eval_genomes(genomes, config, run, stats_data, fitnesses_data, n_nodes_data, n_weights_data, pbar_gens, p.generation)
        winner = p.run(f_fitness, NGEN)

        # Save data
        os.makedirs(game_folder, exist_ok=True)
        with open(os.path.join(game_folder, f'best_individual_run{run}_{ENEMY_MODE}.pkl'), 'wb') as f:
            pickle.dump(winner, f)
                
        df_stats = pd.DataFrame(stats_data)
        df_stats.columns = ['generation', 'max', 'mean', 'std']
        df_stats.to_csv(os.path.join(game_folder, f'stats_run{run}_{ENEMY_MODE}.csv'), index=False)

    np.save(os.path.join(game_folder, f'all_fitnesses_{ENEMY_MODE}.npy'), fitnesses_data)
    np.save(os.path.join(game_folder, f'all_n_nodes_{ENEMY_MODE}.npy'), n_nodes_data)
    np.save(os.path.join(game_folder, f'all_n_weights_{ENEMY_MODE}.npy'), n_weights_data)




if __name__ == '__main__':
    # set up

    set_names, enemy_sets = zip(*ENEMY_SETS.items())
    data_folders = [os.path.join('data', 'neat', f'{set_name}') for set_name in set_names]
    for folder in data_folders:
        os.makedirs(folder, exist_ok=True)

    for name, folder, enemy_set in zip(set_names, data_folders, enemy_sets):
        n_runs = 1
        if '--runs' in sys.argv:
                n_runs = int(sys.argv[sys.argv.index('--runs') + 1])

        n_repeats = 5
        if '--repeats' in sys.argv:
            n_repeats = int(sys.argv[sys.argv.index('--repeats') + 1])

        # do different things depending on the mode
        if '--train' in sys.argv:
            env = create_environment(name, enemy_set, controller_neat)
            run_evolutions(n_runs, name)

        if '--test' in sys.argv:
            gains = np.zeros(n_repeats * n_runs)

            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                CONFIG_PATH)

            for j in range(n_runs):
                with open(os.path.join(DATA_FOLDER, f"{name}/best_individual_run{j}_static.pkl"), 'rb') as f:
                    optimal_weights = pickle.load(f)

                for k in range(n_repeats):
                    net = neat.nn.FeedForwardNetwork.create(optimal_weights, config)
                    _, pl, el, _ = env.play(pcont=net)
                    # Gain = player life - enemy life
                    gains[j*n_repeats + k] = pl - el

            np.save(os.path.join(DATA_FOLDER, f'{name}', 'gains.npy'), gains)

        if '--watch' in sys.argv:
            for run in range(n_runs):
                config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    CONFIG_PATH)
                    
                # Load winner genome
                with open(os.path.join(DATA_FOLDER, str(name), f'best_individual_run{run}_{name}.pkl'), 'rb') as f:
                        winner = pickle.load(f)

                env.update_parameter('visuals', True)
                env.update_parameter('speed', "normal")
                net = neat.nn.FeedForwardNetwork.create(winner, config)
                env.play(pcont=net) # play the game using the best network
        
