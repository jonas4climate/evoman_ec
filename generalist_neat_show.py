import os
import neat
import numpy as np
import pickle

from generalist_shared import create_environment

# Load configuration
from generalist_neat_config import *

np.random.seed(SEED)

def watch(name, enemy_set, folder):
    env = create_environment(name, enemy_set, CONTROLLER)
    for run in range(N_RUNS):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            CONFIG_PATH)
            
        # Load winner genome
        with open(os.path.join(folder, f'best_individual_run{run}_{name}.pkl'), 'rb') as f:
            winner = pickle.load(f)

        env.update_parameter('visuals', True)
        env.update_parameter('speed', "normal")
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        env.play(pcont=net) # play the game using the best network

if __name__ == '__main__':
    # Setup
    set_names, enemy_sets = zip(*ENEMY_SETS.items())
    data_folders = [os.path.join('data', 'neat', f'{set_name}') for set_name in set_names]
    for folder in data_folders:
        os.makedirs(folder, exist_ok=True)

    for name, folder, enemy_set in zip(set_names, data_folders, enemy_sets):
        game_folder = os.path.join(DATA_FOLDER, str(name))
        os.makedirs(game_folder, exist_ok=True)

        watch(name, enemy_set, game_folder)