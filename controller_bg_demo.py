#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# Standard libraries
import sys
import os
import time
import numpy as np

# Evoman framework
### Game environment
from evoman.environment import Environment
### Player's neural network structure
from tutorial.demo_controller import player_controller


EXPERIMENT_NAME = 'optization_bg'
if not os.path.exists(EXPERIMENT_NAME):
    os.makedirs(EXPERIMENT_NAME)

# Update the number of neurons for this specific example
n_hidden_neurons = 10
pcont = player_controller(n_hidden_neurons)

# Initializes environment for single objective mode (specialist) with static enemy and AI player
env = Environment(experiment_name=EXPERIMENT_NAME,
				  playermode="ai",
				  player_controller=pcont,
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)

ENEMIES = range(1,9)
DIFFICULTY = 2

# Tests saved demo solutions for each enemy
for en in ENEMIES:
      # Update the enemy ID
      env.update_parameter('enemies', [en])
  
      # Load specialist controller
      weights = np.load(EXPERIMENT_NAME + f"/ENEMY_{en}_DIFF_{DIFFICULTY}.npy")
      f, _, _, _ = env.play(pcont=weights)

      print(f"Fitness = {f}")
