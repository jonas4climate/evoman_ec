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
    'set_2': [2, 6, 7, 8] # TODO fix
}

GEN_INTERVAL_LOG = 10
NGEN = 50
N_RUNS = 10
N_REPEATS = 1

## Hyperparameter tuning configuration
HP_CONFIG_FILE_NAME = 'best_config.ini'
CONFIG_PATH = os.path.join(DATA_FOLDER, 'set_1', 'hp_crit_mean_of_max', 'best_config.ini') # os.path.join('neat-config-feedforward.ini') # DEFAULT
HP_RANGES = {
    'NEAT.pop_size': (10, 200),
    'DefaultGenome.conn_add_prob': (0.0, 1.0),
    'DefaultGenome.conn_delete_prob': (0.0, 1.0),
    'DefaultGenome.node_add_prob': (0.0, 1.0),  
    'DefaultGenome.node_delete_prob': (0.0, 1.0),
    'DefaultGenome.num_hidden': (1, 10),
    'DefaultGenome.weight_init_mean': (-1.0, 1.0),
    'DefaultGenome.weight_init_stdev': (0.0, 1.0),
    'DefaultGenome.weight_mutate_power': (0.0, 10.0),
    'DefaultGenome.enabled_mutate_rate': (0.0, 1.0),
    'DefaultSpeciesSet.compatibility_threshold': (0.0, 10.0),
}
HP_NGENS = 20 # math.ceil(0.25 * NGEN)
HP_N_RUNS = 3 # Number of runs in hyperparameter tuning to apply criterion for fitness
HP_N_TRIALS = 30 # Number of trials (tuples of hyperparameters to assess) in hyperparameter tuning
HP_PARALLEL_RUNS = 6 # os.cpu_count() # 1 for serial implementation # Number of parallel runs to spawn during tuning
HP_FITNESS_CRITERION = crit_mean_of_max # Fitness criterion applied to fitness matrix. Alternatives defined in util.py

SEED = 42