import os
import numpy as np

from generalist_optimization_test import create_environment, NUM_HIDDEN, N_RUNS
from generalist_optimization_test import EXP_NAME_1, DATA_FOLDER_1
from generalist_optimization_test import EXP_NAME_2, DATA_FOLDER_2

ALL_ENEMIES = range(1, 9)

if __name__ == '__main__':
    # Create environment
    env = create_environment(EXP_NAME_1, ALL_ENEMIES, visuals=True)
    env.speed = 'fastest'

    # Show the best generalist individual for all runs
    for run in range(N_RUNS):

        # Read the optimal weights
        with open(os.path.join(DATA_FOLDER_1, f"best_individual_run{run + 1}.npy"), 'rb') as f:
            optimal_weights = np.load(f)

        # Set the optimal weights
        env.player_controller.set_weights(optimal_weights, NUM_HIDDEN)

        # Play the game
        agg_fit, _, _, _ = env.play()

        # Print the aggregated fitness value:
        print(f"Run #{run}: fit = {agg_fit}")
