import os
from controller_neat import controller_neat

DATA_FOLDER = os.path.join('data', 'neat')
ENEMY_MODE = 'static'
CONFIG_PATH = os.path.join('neat-config-feedforward.ini')

CONTROLLER = controller_neat

# Two groups of enemies
ENEMY_SETS = {
    'set_1': [3, 5, 7],
    'set_2': [2, 6, 7, 8]
}

GEN_INTERVAL_LOG = 10
NGEN = 100 # TODO: make larger in practice
N_RUNS = 5
N_REPEATS = 5

SEED = 42