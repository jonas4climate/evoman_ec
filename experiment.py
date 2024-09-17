################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

from evoman.environment import Environment

from demo_controller import player_controller # Check for neural net inspo
from controller_bartek import controller_bartek
from controller_pablo_francijn import controller_pablo_francijn
from controller_jonas import controller_jonas

experiment_name = 'experiment_1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

chosen_enemies = [1, 2, 3] # pick 3
chosen_controller = controller_bartek()

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=chosen_enemies,
                  multiplemode="yes",
                  enemymode="ai",
                  randomini="no",
                  savelogs="yes",
                  timeexpire=3000,
                  clockprec="low",
                  player_controller=chosen_controller,
                  visuals=True)

env.play()