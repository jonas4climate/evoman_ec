import os
import sys

import neat
import pandas as pd
import numpy as np
import pickle

from evoman.environment import Environment
from controller_pablo_francijn import controller_pablo_francijn

generation = 0
EXP_NAME = 'neat'
DATA_FOLDER = os.path.join('data', EXP_NAME)
ENEMY_MODE = 'static'
ENEMIES = [1] # TODO: make list larger

GEN_INTERVAL_LOG = 10
NGEN = 5 # TODO: make larger in practice

os.makedirs(DATA_FOLDER, exist_ok=True)

for enemy in ENEMIES:

    data = []

    env = Environment(experiment_name=EXP_NAME,
                    enemies=[enemy],
                    multiplemode="no",
                    playermode="ai",
                    player_controller=controller_pablo_francijn(),
                    savelogs="no",
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False
                    )

    def eval_genomes(genomes, config):
        global generation
        
        fitnesses = []
        n_nodes = []
        for id, genome in genomes:
            # create a NN based on the genome provided
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            num_nodes = len(net.node_evals)
            fitness = evaluate_individual(id, net, env) # Calculate fitness based on network's success
            genome.fitness = fitness  # Assign fitness to genome
            fitnesses.append(genome.fitness)
            n_nodes.append(num_nodes)

        max_fitness = max(fitnesses)
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)

        row_data = [generation, max_fitness, mean_fitness, std_fitness, fitnesses]
        data.append(row_data)
        generation += 1

    def evaluate_individual(id, network, env):
        fitness, player_life, enemy_life, time = env.play(pcont=network)
        # adjusting the built-in fitness function 
        fitness_custom = 0.75 * (100 - enemy_life) + 0.25 * player_life - np.log(time)
        return fitness_custom

    def run(config_file):
        
        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)
        
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
    
        p.add_reporter(neat.Checkpointer(generation_interval=GEN_INTERVAL_LOG, time_interval_seconds=None, filename_prefix=os.path.join(DATA_FOLDER, 'neat-checkpoint-')))

        winner = p.run(eval_genomes, NGEN)

        game_folder = os.path.join(DATA_FOLDER, str(enemy))
        os.makedirs(game_folder, exist_ok=True)

        with open(os.path.join(game_folder, f'best_individual_{ENEMY_MODE}.pkl'), 'wb') as f:
            pickle.dump(winner, f)

        # Display the winning genome.
        #print('\nBest genome:\n{!s}'.format(winner))


    if __name__ == '__main__':
        # Determine path to configuration file. 
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'neat-config-feedforward.ini')

        # do different things depending on the mode
        if '--train' in sys.argv:
            run(config_path)
            df = pd.DataFrame(data)
            df.columns = ['generation', 'max', 'mean', 'std', 'fitnesses_individuals']
            game_folder = os.path.join(DATA_FOLDER, str(enemy))
            df.to_csv(os.path.join(game_folder, f'stats_{ENEMY_MODE}.csv'), index=False)

        if '--watch' in sys.argv:
            # Load configuration.
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
            
            # Load winner genome
            with open(os.path.join(DATA_FOLDER, str(enemy), f'best_individual_{ENEMY_MODE}.pkl'), 'rb') as f:
                winner = pickle.load(f)

            env.update_parameter('visuals', True)
            env.update_parameter('speed', "normal")
            net = neat.nn.FeedForwardNetwork.create(winner, config)
            env.play(pcont=net) # play the game using the best network
