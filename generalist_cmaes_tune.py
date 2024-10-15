import random
import os
import multiprocessing
import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler

from deap import base, creator

# Use configuration unless specified otherwise
from generalist_cmaes_config import *
from controller_cmaes import controller_cmaes
from generalist_shared import create_environment
from generalist_cmaes_train import CMAESConfig, run_evolutions

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
optuna.logging.set_verbosity(optuna.logging.WARNING)

np.random.seed(SEED)

def run_trial_in_subprocess(trial, conn, config, set_name, enemy_set, f_criterion=HP_FITNESS_CRITERION):
    """Helper function running for running a single trial of hyperparameter tuning (likely in a separate process). This should not need to be called directly.
    It's primary use is to ensure environment generation occurs in the same processes performing the evolution (pygame limitation).
    
    Args:
        trial (optuna.Trial): optuna trial object for passing trial number and tracking data within optuna
        conn (multiprocessing.Connection): connection to return the fitness value to
        config (CMAESConfig): configuration object for CMA-ES
        set_name (str): name of the environment set
        enemy_set (List[int]): list of enemies to train against
        f_criterion (function, optional): fitness criterion to be used.
    Returns:
        hp_fitness: fitness value for the given trial
    """
    # Custom seeding (NOTE: not thread-safe yet) TODO: make thread-safe with np.random.RandomState passing
    trial_seed = SEED + trial.number
    np.random.seed(trial_seed)
    random.seed(trial_seed)

    # Create environment
    env = create_environment(set_name, enemy_set, controller_cmaes)
    
    # Run evolution
    all_fitnesses, _, _ = run_evolutions(env, config, HP_N_RUNS, pbar_pos=2+trial.number)

    # Calculate fitness and cast into float for optuna
    hp_fitness = float(f_criterion(all_fitnesses))

    # Error handling
    if not isinstance(hp_fitness, (float, int)) or hp_fitness is None:
        raise ValueError(f"Invalid fitness value for trial {trial.number}: {hp_fitness}. Likely a criterion issue.")
    if conn is not None:
        # Return fitness value
        conn.send(hp_fitness)
        conn.close()

def hyperparameter_search(set_name, data_folder, enemy_set):
    """Function for running hyperparameter tuning for CMA-ES on the given environment set.
    By default running separate instances of the game in parallel according to `HP_PARALLEL_RUNS`.

    Args:
        set_name (str): name of the environment set
        data_folder (str): path to the folder where data will be stored
        enemy_set (List[int]): list of enemies to train against
    Returns:
        config: CMAESConfig object with the best hyperparameters
    """

    def run_trial(trial, parallel=False):
        """Wrapper function for running a single trial of hyperparameter tuning.
        Args:
            trial (optuna.Trial): optuna trial object for passing trial number and tracking data within optuna
            parallel (bool, optional): Whether to run the evolution in parallel.
        Returns:
            hp_fitness: fitness value for the given trial
        """
        # Get hyperparameter suggestion from hyperparameter optimizer 
        sigma = trial.suggest_float('sigma', *HP_SIGMA_RANGE, log=True)
        pop_size = trial.suggest_int('pop_size', *HP_POP_SIZE_RANGE)
        config = CMAESConfig(sigma=sigma, population_size=pop_size)

        # Run evolution
        if parallel:
            # Spawn process and set up communication
            parent_conn, child_conn = multiprocessing.Pipe()
            process = multiprocessing.Process(target=run_trial_in_subprocess, args=(trial, child_conn, config, set_name, enemy_set))

            # Run process and receive fitness value
            process.start()
            hp_fitness = parent_conn.recv()
            process.join()

            # Cleanup
            process.close()
            parent_conn.close()
            child_conn.close()
        else:
            # Run trial in the same process
            hp_fitness = run_trial_in_subprocess(trial, None, config, set_name, enemy_set)

        if hp_fitness is None:
            raise RuntimeError(f"Trial {trial.number} failed.")
        return hp_fitness
    
    crit_name = HP_FITNESS_CRITERION.__name__
    parallel = True if HP_PARALLEL_RUNS > 1 else False

    # Run hyperparameter tuning
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: run_trial(trial, parallel), n_trials=HP_N_TRIALS, n_jobs=HP_PARALLEL_RUNS, gc_after_trial=True)

    # Store data
    df = study.trials_dataframe()
    df.to_csv(os.path.join(data_folder, f'hp_trials_data.csv'), index=False)
    best_params = study.best_params
    df = pd.DataFrame(best_params, index=[0])
    df.to_csv(os.path.join(data_folder, f'hp_best_params.csv'), index=False)

    # Get best hyperparameters
    sigma = best_params['sigma']
    population_size = best_params['pop_size']
    config = CMAESConfig(sigma=sigma, population_size=population_size)

    return config

if __name__ == '__main__':
    # Setup
    set_names, enemy_sets = zip(*ENEMY_SETS.items())
    data_folders = [os.path.join('data', 'cmaes', f'{set_name}', f'hp_{HP_FITNESS_CRITERION.__name__}') for set_name in set_names]
    for folder in data_folders:
        os.makedirs(folder, exist_ok=True)
    
    # Core
    for name, folder, enemy_set in zip(set_names, data_folders, enemy_sets):
        print(f'Generating hyperparameter values through tuning for enemy set {name}...\n\n')
        config = hyperparameter_search(name, folder, enemy_set)
        print(f'\n\nBest hyperparameters for {name}: {config}\n\n')
