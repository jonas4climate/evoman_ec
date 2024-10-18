import os
import pickle
import pandas as pd
import numpy as np

from generalist_neat_config import ENEMY_SETS, HP_FITNESS_CRITERION, N_RUNS, CONTROLLER
from generalist_shared import create_environment
from evoman.environment import Environment

# Has to be done this way cause Evoman's Environment consolidates all output values into a single value
def create_singlemode_environment(experiment_name, enemy):
    return Environment(experiment_name=experiment_name,
                enemies=[enemy],
                player_controller=CONTROLLER(),
                savelogs="no",
                logs="off")

if __name__ == '__main__':
    """
    Saves the gain of each individual at each run
    """

    # Setup
    set_names, enemy_sets = zip(*ENEMY_SETS.items())
    data_folders = [os.path.join('data', 'neat', f'{set_name}') for set_name in set_names] # Directory structure is a little different for NEAT folder
#    data_folders = [os.path.join('data', 'neat', f'{set_name}', f'hp_{HP_FITNESS_CRITERION.__name__}') for set_name in set_names]
    for folder in data_folders:
        os.makedirs(folder, exist_ok=True)

    ALL_ENEMIES = range(1,9)
    gains = []

    for i, (name, folder, enemy_set) in enumerate(zip(set_names, data_folders, enemy_sets)):

        for run in range(N_RUNS):

            with open(os.path.join(folder, f"best_individual_run{run}_static.pkl"), 'rb') as f:
                best_individual = pickle.load(f)    # Individual from NEAT is in pickle, not npy

#            run_gain = 0
            run_gains = []

            for enemy in ALL_ENEMIES:
                env = create_singlemode_environment(f"Gain test: {name} run #{run+1}", enemy)
                env.player_controller.set_weights(best_individuals[run], NUM_HIDDEN)

                config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            CONFIG_PATH)
                
                net = neat.nn.FeedForwardNetwork.create(best_individual, config)        # NEAT individual is returned to NN
                _, pl, el, _ = env.play(pcont=net)              

 #               run_gain += pl - el
                run_gains.append(pl - el)


            gains.append(run_gains)

        np.save(os.path.join(folder, "best_gains.npy"), np.array(gains))


