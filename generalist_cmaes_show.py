import os
import numpy as np

# Use configuration unless specified otherwise
from generalist_cmaes_config import *
from generalist_shared import create_environment
from controller_cmaes import controller_cmaes

np.random.seed(SEED)

def get_best_weights(all_fitnesses, best_individuals):
    """Helper function for extracting the best weights across all runs from data returned by `run_evolutions` / stored in files.
    
    Args:
        all_fitnesses (np.array): 3D array with fitness values for each individual in each generation in each run
        best_individuals (np.array): 2D array with best individual from each run
    Returns:
        best_individual: best individual across all runs
    """
    best_run_idx = all_fitnesses.max(axis=(1, 2)).argmax()
    best_individual = best_individuals[best_run_idx]
    return best_individual

if __name__ == '__main__':
    # Setup
    set_names, enemy_sets = zip(*ENEMY_SETS.items())
    data_folders = [os.path.join('data', 'cmaes', f'{set_name}', f'hp_{HP_FITNESS_CRITERION.__name__}') for set_name in set_names]
    
    # Core
    for name, folder, enemy_set in zip(set_names, data_folders, enemy_sets):
        print('Showcasing the best generalist...')
        # Set up environment
        env = create_environment(name, enemy_set, controller_cmaes, visuals=True)
        env.speed = 'normal'

        # Load data and select best weights
        all_fitnesses = np.load(os.path.join(folder, 'train_all_fitnesses.npy'))
        best_individuals = np.load(os.path.join(folder, 'train_best_individuals.npy'))
        best_weights = get_best_weights(all_fitnesses, best_individuals)

        # Set weights and play
        env.player_controller.set_weights(best_weights, NUM_HIDDEN)
        agg_fit, _, _, _ = env.play()
        print(f"Fit = {agg_fit}")
