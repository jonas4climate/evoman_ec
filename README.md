## A Comparison of CMA-ES vs. NEAT on the Evoman video game framework

CMA-ES is a low-maintenance evolutionary strategy that requires little to no fine-tuning to perform numerical optimization but takes in a fixed-size controller architecture. On the other hand, while NEAT is capable of adjusting this architecture to enable higher performance, its large parameter space creates new challenges. In this project, we compare performances of these two algorithms on the Evoman video game playing framework, a testbed for optimization algorithms. A demo can be found [here](https://www.youtube.com/watch?v=ZqaMjd1E4ZI). For more information, please refer to the original repository [here](https://github.com/karinemiras/evoman_framework).

### Setup

To set up the environment, we are using [conda](https://anaconda.org/anaconda/conda). Simply clone this repository, create a local python environment and activate the environment:

```sh
git clone https://github.com/jonas4climate/evoman_ec
conda env -f environment.yaml
conda activate evoman-comp
```


### Experiments

To recreate generation of data and visualize our comparative experiments in the [visualization notebook](./visualization.ipynb), we must run optimization of both algorithms. For this run:

```sh
python optimization_cmaes.py --train --test --runs 10
python optimization_NEAT.py --train --test --runs 10
```

This will run our algorithms with hand-tuned parameter settings on our enemy selection. For changes to this principle, inspect the top of both files. 

You can afterwards also watch the best trained specialists play their respective games by running:

```sh
python optimization_cmaes.py --watch --runs 10
python optimization_NEAT.py --watch --runs 10
```

Feel free to change `--runs` to the number you would like to inspect.

Now you may generate the comparative plots by executing the beforementioned [visualization notebook](./visualization.ipynb) line by line.