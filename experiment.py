################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

from evoman.environment import Environment
from experiment_util import *

from controller_bartek import controller_bartek
from controller_pablo_francijn import controller_pablo_francijn
from controller_jonas import controller_jonas

# Experiment parameters
experiment_name = 'experiment_jonas'
chosen_enemies = [1, 2, 3] # pick 3
chosen_controller = controller_jonas()
n_games = 10

# Generate specialist game environments for chosen controller
envs = []
for enemy in chosen_enemies:
    envs.append(Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  multiplemode="no",
                  enemymode="ai",
                  randomini="no",
                  savelogs="no",
                  timeexpire=3000,
                  clockprec="low",
                  player_controller=chosen_controller,
                  visuals=False))

for env in envs:
    # How to play the game once
    fitness, player_life, enemy_life, game_run_time = env.play()