import os
import neat
import numpy as np
import pickle

from generalist_neat_config import CONFIG_PATH, ENEMY_SETS, HP_FITNESS_CRITERION, N_RUNS, CONTROLLER
from generalist_shared import create_singlemode_environment

if __name__ == '__main__':
    # Setup
    set_names, enemy_sets = zip(*ENEMY_SETS.items())
    data_folders = [os.path.join('data', 'neat', f'{set_name}') for set_name in set_names]
    for folder in data_folders:
        os.makedirs(folder, exist_ok=True)

    ALL_ENEMIES = range(1,9)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH)
    
    for i, (name, folder, enemy_set) in enumerate(zip(set_names, data_folders, enemy_sets)):

        gains = []

        for j in range(N_RUNS):

            run_gain = 0

            # Read optimal weights for this run
            with open(os.path.join(folder, f"best_individual_run{j}_static.pkl"), 'rb') as f:
                optimal_weights = pickle.load(f)

            net = neat.nn.FeedForwardNetwork.create(optimal_weights, config)

            for enemy in ALL_ENEMIES:
                env = create_singlemode_environment(f"Gain test: {name} run #{j+1} enemy {enemy}", enemy, CONTROLLER)

                _, pl, el, _ = env.play(pcont=net)
                run_gain += pl - el

            gains.append(run_gain)
    
        np.save(os.path.join(folder, "best_gains.npy"), np.array(gains))
