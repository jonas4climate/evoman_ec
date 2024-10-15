def crit_mean_of_max(all_fitnesses):
    return all_fitnesses.max(axis=(1, 2)).mean()

def crit_mean(all_fitnesses):
    return all_fitnesses.mean()

def crit_max(all_fitnesses):
    return all_fitnesses.max()