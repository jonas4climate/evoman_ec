import os
import random
import pandas as pd
import numpy as np
import configparser
import neat
import time
import tqdm
import multiprocessing
import optuna
from optuna.samplers import TPESampler

from generalist_shared import create_environment

# Load configuration
from generalist_neat_train import run_evolutions
from generalist_neat_config import *

np.random.seed(SEED)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    env = create_environment(set_name, enemy_set, CONTROLLER)
    
    # Run evolution
    pbar_pos = trial.number % HP_PARALLEL_RUNS
    time.sleep(0.01*trial.number)
    data = run_evolutions(env, config, set_name, pbar_pos, HP_N_RUNS, HP_NGENS, show_output=False, show_tqdm=False)
    all_fitnesses = data[0]

    # Calculate fitness and cast into float for optuna
    hp_fitness = float(f_criterion(all_fitnesses))

    # Error handling
    if not isinstance(hp_fitness, (float, int)) or hp_fitness is None:
        raise ValueError(f"Invalid fitness value for trial {trial.number}: {hp_fitness}. Likely a criterion issue.")
    if conn is not None:
        # Return fitness value
        conn.send(hp_fitness)
        conn.close()

def write_config_to_file(best_params, data_folder):
    config = HP_RANGES.copy()
    for hparam in HP_RANGES.keys():
        section, option = hparam.split('.')
        config[hparam] = best_params[option]

    print(f'\n\nBest hyperparameter subset for {name}: {config}\n\n')

    config_parser = configparser.ConfigParser()
    config_parser.read(CONFIG_PATH)

    for hparam, value in config.items():
        section, option = hparam.split('.')
        if section not in config_parser:
            config_parser.add_section(section)
        config_parser.set(section, option, str(value))

    with open(os.path.join(data_folder, HP_CONFIG_FILE_NAME), 'w') as f:
        config_parser.write(f)

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

    pbar = tqdm.tqdm(total=HP_N_TRIALS, desc=f'Hyperparameter tuning for {set_name}', unit='trial')

    def run_trial(trial, pbar, parallel=False):
        """Wrapper function for running a single trial of hyperparameter tuning.
        Args:
            trial (optuna.Trial): optuna trial object for passing trial number and tracking data within optuna
            parallel (bool, optional): Whether to run the evolution in parallel.
        Returns:
            hp_fitness: fitness value for the given trial
        """
        trial_hp_dict = HP_RANGES.copy()
        # Get hyperparameter suggestion from hyperparameter optimizer 
        for hparam, (low, high) in HP_RANGES.items():
            section, option = hparam.split('.')
            if isinstance(low, int) and isinstance(high, int):
                trial_hp_dict[option] = trial.suggest_int(option, low, high)
            elif isinstance(low, float) and isinstance(high, float):
                trial_hp_dict[option] = trial.suggest_float(option, low, high)
            else:
                raise ValueError(f"Invalid hyperparameter types for {hparam}: {low}, {high} are unsupported")
            
        # Load base configuration
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            CONFIG_PATH)
        
        # Overwrite with hyperparameter trial values
        for key, value in trial_hp_dict.items():
            setattr(config, key, value)

        # Run evolution
        if parallel:
            # Spawn process and set up communication
            parent_conn, child_conn = multiprocessing.Pipe()
            process = multiprocessing.Process(target=run_trial_in_subprocess, args=(trial, child_conn, config, set_name, enemy_set))

            # Run process and receive fitness value
            process.start()
            hp_fitness = parent_conn.recv()
            process.join()
            pbar.update(1)
            print(f"Trial {trial.number} completed with fitness {hp_fitness}")

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
    study.optimize(lambda trial: run_trial(trial, pbar, parallel), n_trials=HP_N_TRIALS, n_jobs=HP_PARALLEL_RUNS, gc_after_trial=True)

    # Store data
    print(f"Saving hyperparameter tuning data to {data_folder}...") 
    df = study.trials_dataframe()
    df.to_csv(os.path.join(data_folder, f'hp_trials_data.csv'), index=False)
    best_params = study.best_params
    df = pd.DataFrame(best_params, index=[0])
    df.to_csv(os.path.join(data_folder, f'hp_best_params.csv'), index=False)

    # Save the best hyperparameters to a .ini file
    write_config_to_file(best_params, data_folder)

if __name__ == '__main__':
    # Setup
    set_names, enemy_sets = zip(*ENEMY_SETS.items())
    data_folders = [os.path.join('data', 'neat', f'{set_name}', f'hp_{HP_FITNESS_CRITERION.__name__}') for set_name in set_names]
    for folder in data_folders:
        os.makedirs(folder, exist_ok=True)
    
    # Core
    for name, folder, enemy_set in zip(set_names, data_folders, enemy_sets):
        print(f'Generating hyperparameter values through tuning for enemy set {name}...\n\n')
        hyperparameter_search(name, folder, enemy_set)
