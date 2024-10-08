import os
import numpy as np

from generalist_optimization_test import create_environment, NUM_HIDDEN, N_RUNS
from generalist_optimization_test import EXP_NAME_1, DATA_FOLDER_1, ENEMY_SET_1
from generalist_optimization_test import EXP_NAME_2, DATA_FOLDER_2, ENEMY_SET_2

ALL_ENEMIES = range(1, 9)

if __name__ == '__main__':

    for (exp_name, data_folder, enemy_set) in zip([EXP_NAME_1, EXP_NAME_2], [DATA_FOLDER_1, DATA_FOLDER_2], [ENEMY_SET_1, ENEMY_SET_2]):

        # Create environment
        env = create_environment(exp_name, ALL_ENEMIES, visuals=True)
        env.speed = 'fastest'

        # Show which enemies the generalist was trained on
        print(f"\n=====| Trained on enemy set: {enemy_set} |=====")

        # Determine run with best individual
        with open(os.path.join(data_folder, f"all_fitnesses.npy"), 'rb') as f:
            all_fitnesses = np.load(f)
        best_run = np.argmax(np.max(all_fitnesses, axis=(1,2))) + 1

        # Read weights of the best generalist individual for all runs
        with open(os.path.join(data_folder, f"best_individual_run{best_run}.npy"), 'rb') as f:
            optimal_weights = np.load(f)

        # Set the optimal weights
        env.player_controller.set_weights(optimal_weights, NUM_HIDDEN)

        # Play the game
        agg_fit, _, _, _ = env.play()

        # Print the aggregated fitness value:
        print(f"Best run #{best_run}: fit = {agg_fit}")
