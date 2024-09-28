import neat
import os
import numpy as np
import pickle

from tqdm import tqdm
from optimization_NEAT import ENEMIES, DATA_FOLDER
from evoman.environment import Environment
from EC.evoman_ec.controller_neat import controller_neat

# Number of repeated games played per optimal individual in a run
N_REPEATS = 5

# Number of runs per algorithm, per enemy
N_RUNS = 10

# Config for running NEAT module
CONFIG_PATH = os.path.join('.', 'neat-config-feedforward.ini')
CONFIG = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH)

env = Environment(experiment_name='data/neat', multiplemode="no", playermode="ai", player_controller=controller_neat(),
                                savelogs="no", logs='off', enemymode="static", level=2, speed="fastest", visuals=False)

if __name__ == '__main__':
    for (i, enemy) in tqdm(enumerate(ENEMIES), total=len(ENEMIES)):
        gains = np.zeros(N_REPEATS * N_RUNS)

        for j in tqdm(range(N_RUNS), leave=False):
            with open(os.path.join(DATA_FOLDER, f"{enemy}/best_individual_run{j}_static.pkl"), 'rb') as f:
                neat_optimal_weights = pickle.load(f)

            for k in range(N_REPEATS):
                # Environment & Initial network
                net = neat.nn.FeedForwardNetwork.create(neat_optimal_weights, CONFIG)
                env.update_parameter('enemies', [enemy])
                
                # Play the game
                _, pl, el, _ = env.play(pcont=net)
                
                # Gain = player life - enemy life
                gains[j*N_REPEATS + k] = pl - el

        # Save the gains
        np.save(os.path.join(DATA_FOLDER, f'{enemy}', 'gains.npy'), gains)