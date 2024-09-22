import os

import neat
import visualize
import numpy as np

from evoman.environment import Environment
from controller_pablo_francijn import controller_pablo_francijn

dom_u = 1
dom_l = -1
npop = 100
experiment_name = "NEAT"
enemy_list = range(1, 9)


# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  enemies=enemy_list,
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=controller_pablo_francijn(),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False
                  )

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # create a NN based on the genome provided
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        fitness += evaluate_individual(net, env) # Calculate fitness based on network's success
        
        genome.fitness = fitness  # Assign fitness to genome


def evaluate_individual(action, env):
    _, player_life, enemy_life, time = env.play(pcont=action)
    fitness = 0.75 * (100 - enemy_life) + 0.25 * player_life - np.log(time)
    return fitness

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
   
    # Checkpoint saving every 10 generations
    p.add_reporter(neat.Checkpointer(generation_interval=10, time_interval_seconds=None))

    # Run the evolution for up to 30 generations
    winner = p.run(eval_genomes, 30)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    #config_path = r"C:\Users\franc\Documents\GitHub\evoman_ec\config-feedforward"
    #print(os.path.isfile(config_path))  # Should return True if the file exists
    local_dir = os.path.dirname(__file__)
    config_path = r"C:\Users\franc\Documents\GitHub\evoman_ec\config-feedforward.ini"
    run(config_path)


    