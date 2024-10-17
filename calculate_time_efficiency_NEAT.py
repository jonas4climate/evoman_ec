import numpy as np
import pandas as pd
import os 

# Load configuration
from generalist_neat_config import *


# SET 2 STILL NEEDS TO BE ADDED TO THIS LIST BUT DO NOT HAVE DATA ATM
for set in ENEMY_SETS.keys():
    time_efficiency_runs = np.zeros((N_RUNS, NGEN))
    for run in range(N_RUNS):
        path = f'{DATA_FOLDER}\\{set}\\'
        df = pd.read_csv(f'{path}stats_run{run}_static.csv')
        time_difference = []
        max_fitness_difference = []
        df["prev_time"] = df["time"].shift(1)
        df["prev_max_fitness"] = df["max"].shift(1)

        for i in range(len(df["time"])):                                # for each generation 
            time_difference.append(df["prev_time"][i] - df["time"][i])
            max_fitness_difference.append(df["prev_max_fitness"][i] - df["max"][i])

        time_difference = np.array(time_difference)
        max_fitness_difference = np.array(max_fitness_difference)

        time_efficiency = max_fitness_difference/time_difference
        time_efficiency_runs[run] = time_efficiency
    
    np.save(os.path.join(path, f'time_efficiency_{set}.npy'), time_efficiency_runs)

    
