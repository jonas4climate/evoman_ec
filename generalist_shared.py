from evoman.environment import Environment
import numpy as np

def create_environment(experiment_name, enemy_set, controller, visuals=False):
    """Returns an Environment object for the Evoman framework.
    Args:
        experiment_name (string): name of the experiment, used for logging
        enemy_set (List[int]): list with enemies to train on
        visuals (bool, optional): Whether to display environment to the screen. Defaults to False.
    Returns:
        Environment: Environment object representing the the Evoman instance
    """
    env = Environment(experiment_name=experiment_name,
                        enemies=enemy_set,
                        multiplemode="yes",
                        enemymode='static',
                        speed='normal' if visuals else 'fastest',
                        player_controller=controller(),
                        savelogs="no",
                        logs="off",
                        clockprec="low",
                        visuals=visuals)
    return env

# Has to be done this way cause Evoman's Environment consolidates all output values into a single value
def create_singlemode_environment(experiment_name, enemy, controller):
    return Environment(experiment_name=experiment_name,
                enemies=[enemy],
                player_controller=controller(),
                savelogs="no",
                logs="off")