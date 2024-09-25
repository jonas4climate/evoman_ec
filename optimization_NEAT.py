import os
import sys

import neat
import pandas as pd
import numpy as np
import pickle
import tqdm

from evoman.environment import Environment
from controller_pablo_francijn import controller_pablo_francijn

EXP_NAME = 'neat'
DATA_FOLDER = os.path.join('data', EXP_NAME)
ENEMY_MODE = 'static'
ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8] # TODO: make list larger

# Custom fitness parameters
FITNESS_GAMMA = 0.75
FITNESS_ALPHA = 0.25

GEN_INTERVAL_LOG = 10
NGEN = 50 # TODO: make larger in practice

SEED = 42
np.random.seed(SEED)

os.makedirs(DATA_FOLDER, exist_ok=True)

class FixedSizeReproduction(neat.DefaultReproduction):
    """Provides a reproduction class that ensures the population size stays fixed."""
    def reproduce(self, config, species, pop_size, generation):
        # Call the default reproduce method
        offspring = super().reproduce(config, species, pop_size, generation)
        
        # Ensure population size stays fixed by truncating or adding genomes as needed
        if len(offspring) > pop_size:
            # Truncate the offspring to the desired population size
            offspring = dict(list(offspring.items())[:pop_size])
        elif len(offspring) < pop_size:
            # Add random genomes to meet the population size requirement
            while len(offspring) < pop_size:
                new_id = max(offspring.keys()) + 1
                # Create a random new genome to fill the population
                offspring[new_id] = neat.DefaultGenome(new_id)
        
        return offspring
    
pbar2 = tqdm.tqdm(total=len(ENEMIES), desc='Training on games', unit='game', position=0)

for enemy in ENEMIES:

    generation = 0

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

    def eval_genomes(genomes, config, run, stats_data, fitnesses_data, n_nodes_data, pbar):
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

        n_nodes_data[run, generation, :len(n_nodes)] = n_nodes
        fitnesses_data[run, generation, :len(fitnesses)] = fitnesses
        max_fitness = max(fitnesses)
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)

        row_data = [generation, max_fitness, mean_fitness, std_fitness]
        stats_data.append(row_data)
        generation += 1
        pbar.update(1)

    def evaluate_individual(id, network, env):
        _, player_life, enemy_life, time = env.play(pcont=network)
        # adjusting the built-in fitness function 
        fitness_custom = FITNESS_ALPHA * (100 - enemy_life) + FITNESS_GAMMA * player_life - np.log(time)
        return fitness_custom

    def run_evolutions(config_file, n_runs=1):
        # Find out pop size
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)
        pop_size = int(config.pop_size * 1.1) # 10% margin for growth
        fitnesses_data = np.full((n_runs, NGEN, pop_size), np.nan)
        n_nodes_data = np.full((n_runs, NGEN, pop_size), np.nan)

        game_folder = os.path.join(DATA_FOLDER, str(enemy))
        pbar = tqdm.tqdm(total=n_runs*NGEN, desc=f'Training specialist against enemy {enemy}', unit='gen', position=1)

        for run in range(n_runs):
            global generation
            generation = 0

            stats_data = []

            # Load configuration.
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
            
            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(config)

            # Add a stdout reporter to show progress in the terminal.
            # p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            # p.add_reporter(neat.Checkpointer(generation_interval=GEN_INTERVAL_LOG, time_interval_seconds=None, filename_prefix=os.path.join(DATA_FOLDER, 'neat-checkpoint-')))

            winner = p.run(lambda genomes, config: eval_genomes(genomes, config, run, stats_data, fitnesses_data, n_nodes_data, pbar), NGEN)

            # Save data
            os.makedirs(game_folder, exist_ok=True)
            with open(os.path.join(game_folder, f'best_individual_run{run}_{ENEMY_MODE}.pkl'), 'wb') as f:
                pickle.dump(winner, f)
                
            df_stats = pd.DataFrame(stats_data)
            df_stats.columns = ['generation', 'max', 'mean', 'std']
            df_stats.to_csv(os.path.join(game_folder, f'stats_run{run}_{ENEMY_MODE}.csv'), index=False)

        np.save(os.path.join(game_folder, f'all_fitnesses_{ENEMY_MODE}.npy'), fitnesses_data)
        np.save(os.path.join(game_folder, f'all_n_nodes_{ENEMY_MODE}.npy'), n_nodes_data)


    if __name__ == '__main__':
        # Determine path to configuration file. 
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'neat-config-feedforward.ini')

        # do different things depending on the mode
        if '--train' in sys.argv:
            runs = 1
            if '--runs' in sys.argv:
                runs = int(sys.argv[sys.argv.index('--runs') + 1])
            run_evolutions(config_path, runs)
            pbar2.update(1)

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
    
