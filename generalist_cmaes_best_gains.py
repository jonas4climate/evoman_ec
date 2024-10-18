import os
import numpy as np

from generalist_cmaes_config import ENEMY_SETS, HP_FITNESS_CRITERION, N_RUNS, CONTROLLER, NUM_HIDDEN
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
    # Setup
    set_names, enemy_sets = zip(*ENEMY_SETS.items())
    data_folders = [os.path.join('data', 'cmaes', f'{set_name}', f'hp_{HP_FITNESS_CRITERION.__name__}') for set_name in set_names]
    for folder in data_folders:
        os.makedirs(folder, exist_ok=True)

    ALL_ENEMIES = range(1,9)

    for i, (name, folder, enemy_set) in enumerate(zip(set_names, data_folders, enemy_sets)):

        best_individuals = np.load(os.path.join(folder, "train_best_individuals.npy"))
        gains = []

        for run in range(N_RUNS):

            run_gain = 0

            for enemy in ALL_ENEMIES:
                env = create_singlemode_environment(f"Gain test: {name} run #{run+1}", enemy)
                env.player_controller.set_weights(best_individuals[run], NUM_HIDDEN)

                _, pl, el, _ = env.play()
                run_gain += pl - el

            gains.append(run_gain)

        np.save(os.path.join(folder, "best_gains.npy"), np.array(gains))


