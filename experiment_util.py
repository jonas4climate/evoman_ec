################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
from evoman.environment import Environment
from evoman.controller import Controller

def watch_controller_play(envs: list, n_games=1, speed="normal"):
    for env in envs:
        env.visuals = True
        env.speed = speed
        for _ in range(n_games):
            env.play()
