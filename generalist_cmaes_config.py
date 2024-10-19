## Configuration file

from criterions import * 
from controller_cmaes import controller_cmaes
from demo_controller import player_controller
import os

# Two groups of enemies
ENEMY_SETS = {
    'set_1': [3, 5, 7],
    'set_2': [2, 6, 7, 8]
}

## Training configuration
N_RUNS = 10 # Number of repeated runs in training

## Hyperparameter values for CMA-ES
HP_LOAD_FROM_FILE = True # If True, ignores hardcoded values below
POPULATION_SIZE = 58 # (tune-able)
SIGMA = 0.7336350010265406 # (tune-able)
NGEN = 50 # (not tune-able)

## Hyperparameter tuning configuration
HP_POP_SIZE_RANGE = (10, 200) # Tuning range for population size
HP_SIGMA_RANGE = (0.1, 10.0) # Tuning range for sigma
HP_N_RUNS = 3 # Number of runs in hyperparameter tuning to apply criterion for fitness
HP_N_TRIALS = 10 # Number of trials (tuples of hyperparameters to assess) in hyperparameter tuning
HP_PARALLEL_RUNS = 6 # os.cpu_count() # 1 for serial implementation # Number of parallel runs to spawn during tuning
HP_FITNESS_CRITERION = crit_mean # Fitness criterion applied to fitness matrix. Alternatives defined in util.py

## Controller parameters
NUM_HIDDEN = 10  # Number of hidden layer neurons
CONTROLLER = controller_cmaes

## Seed for reproducibility
SEED = 42

## Logging
HOF_SIZE = 1 # Hall of fame size to store. NOTE: currently only 1 is supported