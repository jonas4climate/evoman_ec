import os
import numpy as np
import pickle

from tqdm import tqdm
from optimization_cmaes import ENEMIES, DATA_FOLDER, NUM_HIDDEN
from evoman.environment import Environment
from controller_jonas import controller_jonas

# Number of repeated games played per optimal individual in a run
N_REPEATS = 5

# Number of runs per algorithm, per enemy
N_RUNS = 10

env = Environment(experiment_name='data/neat', multiplemode="no", playermode="ai", player_controller=controller_jonas(),
                                savelogs="no", logs='off', enemymode="static", level=2, speed="fastest", visuals=False)

if __name__ == '__main__':
    for (i, enemy) in tqdm(enumerate(ENEMIES), total=len(ENEMIES)):
        gains = np.zeros(N_REPEATS * N_RUNS)

        for j in tqdm(range(N_RUNS), leave=False):
            with open(os.path.join(DATA_FOLDER, f"{enemy}/best_individual_run{j}_static.npy"), 'rb') as f:
                neat_optimal_weights = np.load(f)

            # Set the network weights
            env.player_controller.set_weights(neat_optimal_weights, NUM_HIDDEN)

            for k in range(N_REPEATS):
                # Environment
                env.update_parameter('enemies', [enemy])
                
                # Play the game
                _, pl, el, _ = env.play()
                
                # Gain = player life - enemy life
                gains[j*N_REPEATS + k] = pl - el

        # Save the gains
        np.save(os.path.join(DATA_FOLDER, f'{enemy}', 'gains.npy'), gains)