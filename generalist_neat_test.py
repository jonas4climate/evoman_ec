import os
import neat
import numpy as np
import pickle

from generalist_shared import create_environment

# Load configuration
from generalist_neat_config import *

np.random.seed(SEED)

def test(name, folder, enemy_set):
    all_enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    env = create_environment(name, all_enemies, CONTROLLER)
    gains = np.zeros(N_REPEATS * N_RUNS)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        CONFIG_PATH)

    for j in range(N_RUNS):
        with open(os.path.join(folder, f"best_individual_run{j}_static.pkl"), 'rb') as f:
            optimal_weights = pickle.load(f)

        for k in range(N_REPEATS):
            net = neat.nn.FeedForwardNetwork.create(optimal_weights, config)
            _, pl, el, _ = env.play(pcont=net)
            # Gain = player life - enemy life
            gains[j*N_REPEATS + k] = pl - el

    np.save(os.path.join(folder, 'gains.npy'), gains)

    return gains

if __name__ == '__main__':
    # Setup
    set_names, enemy_sets = zip(*ENEMY_SETS.items())
    data_folders = [os.path.join('data', 'neat', f'{set_name}') for set_name in set_names]
    for folder in data_folders:
        os.makedirs(folder, exist_ok=True)

    for name, folder, enemy_set in zip(set_names, data_folders, enemy_sets):
        game_folder = os.path.join(DATA_FOLDER, str(name))
        os.makedirs(game_folder, exist_ok=True)

        test(name, folder, enemy_set)