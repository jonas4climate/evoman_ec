## A Comparison of CMA-ES vs. NEAT on the Evoman video game framework 

CMA-ES is a low-maintenance evolutionary strategy that requires little to no fine-tuning to perform numerical optimization but takes in a fixed-size controller architecture. On the other hand, while NEAT is capable of adjusting this architecture to enable higher performance, its large parameter space creates new challenges. In this project, we compare performances of these two algorithms on the Evoman video game playing framework, a testbed for optimization algorithms. A demo can be found [here](https://www.youtube.com/watch?v=ZqaMjd1E4ZI). For more information, please refer to the original repository [here](https://github.com/karinemiras/evoman_framework).

## Task I and Task II
The project consisted of two parts: training a specialist agent in Task I and training a generalis agent in Task II. Furthermore, in Task I we looked into the differences between the actions taken by the fittest agent of both algorithms, while in Task II we were interested in the difference in computational efficiency of the two algorithms.

### Setup

To set up the environment, we are using [conda](https://anaconda.org/anaconda/conda). Simply clone this repository, create a local python environment and activate the environment:

```sh
git clone https://github.com/jonas4climate/evoman_ec
conda env create -f environment.yaml
conda activate evoman-comp
```


### Experiments Task I

To recreate generation of data and visualize our comparative experiments in the [specialist visualization notebook](./visualization.ipynb), we must run optimization of both algorithms. For this run:

```sh
python specialist_optimization_cmaes.py --train --test --runs 10
python specialist_optimization_NEAT.py --train --test --runs 10
```

This will run our algorithms with hand-tuned parameter settings on our enemy selection. For changes to this principle, inspect the top of both files. 

You can afterwards also watch the best trained specialists play their respective games by running:

```sh
python specialist_optimization_cmaes.py --watch --runs 10
python specialist_optimization_NEAT.py --watch --runs 10
```

Feel free to change `--runs` to the number you would like to inspect.

Now you may generate the comparative plots by executing the beforementioned [specialist visualization notebook](./visualization.ipynb) line by line.

### Report Task I

The report summarizing our findings can be found [here](./105.pdf).

### Experiments Task II
To recreate generation of data and visualize our comparative experiments in the [generalist visualization notebook](./visualization.ipynb), we first must run hyperparameter tuning for both algorithms. For this, set parameter ranges in generalist_cmaes_config.py and generalist_neat_config.py and run:

```sh
python generalist_cmaes_tune.py
python generalist_neat_tune.py
```

Then we must run optimization for both algorithms:

```sh
python generalist_cmaes_train.py
python generalist_neat_train.py
```
Then to obtain data, we must run the following:
```sh
python generalist_cmaes_best_gains.py
python generalist_neat_test.py
```

You can afterwards also watch the best trained generalist play against all the enemies by running:

```sh
python generalist_cmaes_show.py
python generalist_neat_show.py
```

Now you may generate the comparative plots by executing the beforementioned [generalist visualization notebook](./visualization.ipynb) line by line.

