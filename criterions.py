import numpy as np

def crit_mean_of_max(all_fitnesses):
    return np.nanmean(np.nanmax(all_fitnesses, axis=(1, 2)))

def crit_mean(all_fitnesses):
    return np.nanmean(all_fitnesses)

def crit_max(all_fitnesses):
    return np.nanmax(all_fitnesses)