import os
import sys

import neat
import pandas as pd
import numpy as np
import pickle
import tqdm

from evoman.environment import Environment
from controller_neat import controller_neat

EXP_NAME = 'neat'
DATA_FOLDER = os.path.join('data', EXP_NAME)
ENEMY_MODE = 'static'
CONFIG_PATH = os.path.join('neat-config-feedforward.ini')
ENEMIES = [1, 3, 4] # TODO: make list larger

# Custom fitness parameters
FITNESS_GAMMA = 0.75
FITNESS_ALPHA = 0.25

GEN_INTERVAL_LOG = 10
NGEN = 100 # TODO: make larger in practice

SEED = 42
np.random.seed(SEED)

os.makedirs(DATA_FOLDER, exist_ok=True)
    
pbar_games = tqdm.tqdm(total=len(ENEMIES), desc='Training on games', unit='game', position=0)

for (i, enemy) in enumerate(ENEMIES):

    generation = 0

    env = Environment(experiment_name=EXP_NAME,
                    enemies=[enemy],
                    multiplemode="no",
                    playermode="ai",
                    player_controller=controller_neat(),
                    savelogs="no",
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False
                    )

    def eval_genomes(genomes, config, run, stats_data, fitnesses_data, n_nodes_data, all_decisions, n_weights_data, pbar_gens):
        global generation
        
        fitnesses = []
        n_nodes = []
        n_weights = []
        for id, genome in genomes:
            # create a NN based on the genome provided
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            num_nodes = len(net.node_evals)
            fitness = evaluate_individual(id, net, env) # Calculate fitness based on network's success
            genome.fitness = fitness  # Assign fitness to genome
            fitnesses.append(genome.fitness)
            n_weights.append(len(genome.connections))
            n_nodes.append(num_nodes)

        n_nodes_data[run, generation, :len(n_nodes)] = [np.mean(n_nodes[i]) for i in range(len(n_nodes))]
        n_weights_data[run, generation, :len(n_weights)] = [np.mean(n_weights[i]) for i in range(len(n_weights))]
        fitnesses_data[run, generation, :len(fitnesses)] = fitnesses
        mean_number_nodes = np.mean(n_nodes)
        max_fitness = max(fitnesses)
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)

        decision_history = np.array(env.player_controller.decision_history)
        dec_mean, dec_std = np.mean(decision_history, axis=0), np.std(decision_history, axis=0)
        all_decisions[run, generation] = [dec_mean, dec_std]
        env.player_controller.reset_history()

        row_data = [generation, max_fitness, mean_fitness, std_fitness, mean_number_nodes]
        stats_data.append(row_data)
        generation += 1
        pbar_gens.update(1)

    def evaluate_individual(id, network, env):
        _, player_life, enemy_life, time = env.play(pcont=network)
        # adjusting the built-in fitness function 
        fitness_custom = FITNESS_GAMMA * (100 - enemy_life) + FITNESS_ALPHA * player_life - np.log(time)
        return fitness_custom

    def run_evolutions(n_runs=1):
        # Find out pop size
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            CONFIG_PATH)
        pop_size = int(config.pop_size * 1.1) # 10% margin for growth
        fitnesses_data = np.full((n_runs, NGEN, pop_size), np.nan)
        n_nodes_data = np.full((n_runs, NGEN, pop_size), np.nan)
        n_weights_data = np.full((n_runs, NGEN, pop_size), np.nan)
        # decision format stored: [left, right, jump, shoot, release, left_or_right, jump_or_release]
        all_decisions = np.zeros((n_runs, NGEN, 2, 7)) # mean and std for each type of decision at each generation & run

        game_folder = os.path.join(DATA_FOLDER, str(enemy))
        pbar_gens = tqdm.tqdm(total=n_runs*NGEN, desc=f'Training specialist against enemy {enemy}', unit='gen', position=1)

        for run in range(n_runs):
            global generation
            generation = 0

            stats_data = []

            # Load configuration.
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                CONFIG_PATH)
            
            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(config)

            # Add a stdout reporter to show progress in the terminal.
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            # p.add_reporter(neat.StdOutReporter(True))
            # p.add_reporter(neat.Checkpointer(generation_interval=GEN_INTERVAL_LOG, time_interval_seconds=None, filename_prefix=os.path.join(DATA_FOLDER, 'neat-checkpoint-')))

            winner = p.run(lambda genomes, config: eval_genomes(genomes, config, run, stats_data, fitnesses_data, n_nodes_data, all_decisions, n_weights_data, pbar_gens), NGEN)

            # Save data
            os.makedirs(game_folder, exist_ok=True)
            with open(os.path.join(game_folder, f'best_individual_run{run}_{ENEMY_MODE}.pkl'), 'wb') as f:
                pickle.dump(winner, f)
                
            df_stats = pd.DataFrame(stats_data)
            df_stats.columns = ['generation', 'max', 'mean', 'std', 'mean no. nodes']
            df_stats.to_csv(os.path.join(game_folder, f'stats_run{run}_{ENEMY_MODE}.csv'), index=False)

        np.save(os.path.join(game_folder, f'all_fitnesses_{ENEMY_MODE}.npy'), fitnesses_data)
        np.save(os.path.join(game_folder, f'all_n_nodes_{ENEMY_MODE}.npy'), n_nodes_data)
        np.save(os.path.join(game_folder, f'all_n_weights_{ENEMY_MODE}.npy'), n_weights_data)
        np.save(os.path.join(game_folder, f'all_decisions_{ENEMY_MODE}.npy'), all_decisions)


    if __name__ == '__main__':

        n_runs = 1
        if '--runs' in sys.argv:
            n_runs = int(sys.argv[sys.argv.index('--runs') + 1])

        n_repeats = 5
        if '--repeats' in sys.argv:
            n_repeats = int(sys.argv[sys.argv.index('--repeats') + 1])

        # do different things depending on the mode
        if '--train' in sys.argv:
            run_evolutions(n_runs)
            pbar_games.update(1)

        if '--test' in sys.argv:
            gains = np.zeros(n_repeats * n_runs)

            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            CONFIG_PATH)

            for j in range(n_runs):
                with open(os.path.join(DATA_FOLDER, f"{enemy}/best_individual_run{j}_static.pkl"), 'rb') as f:
                    optimal_weights = pickle.load(f)

                for k in range(n_repeats):
                    net = neat.nn.FeedForwardNetwork.create(optimal_weights, config)
                    _, pl, el, _ = env.play(pcont=net)
                    # Gain = player life - enemy life
                    gains[j*n_repeats + k] = pl - el

            np.save(os.path.join(DATA_FOLDER, f'{enemy}', 'gains.npy'), gains)

        if '--watch' in sys.argv:
            for run in range(n_runs):
                config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                CONFIG_PATH)
                
                # Load winner genome
                with open(os.path.join(DATA_FOLDER, str(enemy), f'best_individual_run{run}_{ENEMY_MODE}.pkl'), 'rb') as f:
                    winner = pickle.load(f)

                env.update_parameter('visuals', True)
                env.update_parameter('speed', "normal")
                net = neat.nn.FeedForwardNetwork.create(winner, config)
                env.play(pcont=net) # play the game using the best network
    
