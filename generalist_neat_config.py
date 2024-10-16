import os
import math
from criterions import *
from controller_neat import controller_neat

DATA_FOLDER = os.path.join('data', 'neat')
ENEMY_MODE = 'static'

CONTROLLER = controller_neat

# Two groups of enemies
ENEMY_SETS = {
    'set_1': [3, 5, 7],
    # 'set_2': [2, 6, 7, 8] # TODO fix
}

GEN_INTERVAL_LOG = 10
NGEN = 20 # TODO: make larger in practice
N_RUNS = 2 # TODO change
N_REPEATS = 5

## Hyperparameter tuning configuration
HP_CONFIG_FILE_NAME = 'best_config.ini'
CONFIG_PATH = os.path.join('neat-config-feedforward.ini')
HP_RANGES = {
    'NEAT.pop_size': (10, 200),
    'DefaultGenome.conn_add_prob': (0.0, 1.0),
    'DefaultGenome.conn_delete_prob': (0.0, 1.0),
    'DefaultGenome.node_add_prob': (0.0, 1.0),  
    'DefaultGenome.node_delete_prob': (0.0, 1.0)
}
HP_NGENS = 1 # math.ceil(0.25 * NGEN)
HP_N_RUNS = 2 # 3 # Number of runs in hyperparameter tuning to apply criterion for fitness
HP_N_TRIALS = 2 # 30 # Number of trials (tuples of hyperparameters to assess) in hyperparameter tuning
HP_PARALLEL_RUNS = os.cpu_count() # 1 for serial implementation # Number of parallel runs to spawn during tuning
HP_FITNESS_CRITERION = crit_mean_of_max # Fitness criterion applied to fitness matrix. Alternatives defined in util.py

SEED = 42