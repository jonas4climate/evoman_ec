################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
from tqdm import tqdm
import pandas as pd

from evoman.environment import Environment

from demo_controller import player_controller # Check for neural net inspo
from controller_bartek import controller_bartek
from controller_pablo_francijn import controller_pablo_francijn
from controller_jonas import controller_jonas

def generate_controller_eval_data(envs: list, n_games: int, experiment_name: str):
    for env in envs:
        path = f"data/{experiment_name}/{env.enemies}"
        os.makedirs(path, exist_ok=True)
        data = []
        for _ in tqdm(range(n_games)):
            fitness, player_life, enemy_life, game_run_time = env.play()
            data.append({
                'fitness': fitness,
                'player_life': player_life,
                'enemy_life': enemy_life,
                'game_run_time': game_run_time
            })

        metrics = pd.DataFrame(data)
        metrics.to_csv(f"{path}/metrics.csv", index=False)

def watch_controller_play(envs: list, n_games=1, speed="normal"):
    for env in envs:
        env.visuals = True
        env.speed = speed
        for _ in range(n_games):
            env.play()