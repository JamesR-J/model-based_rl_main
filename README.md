# Jax Model-Based RL 

The beginnings of a benchmark suite of Model-Based Reinforcement Learning (RL) approaches based in Jax. 
We aim to provide a consistent location for many of the SOTA methods today, enabling much easier benchmarking of new algorithmic improvements as well as provide a useful testbed for practitioners.
We are very open to suggestions and contributions so please get in touch!


### Why Jax?

Jax has become almost a staple in Model-Free RL due to the extreme potential of computational speed-ups, in part due to the vectorisation of environments enabling both RL agent and environment to be run together on a GPU. This is great for many applications of Model-Based RL as they are generally continuous control domains that rely on differentiable physics; another domain that has seen the rapid adoption of Jax due to flexible GPU and batch possibilties alongside the powerful gradient calculations. Mujoco has already been rewritten in Jax at the [Brax](https://github.com/google/brax) package.

However, there are some challenges in its adoption for Model-Based RL. When using traced variables, dataset sizes must be consistent, an issue when tracking a continuously updating dataset during model training. Model-Free methods can avoid this as set size samples are taken from replay buffers for off-policy methods, or consistent rollout lengths are used for on-policy approaches. Nonetheless, each step in the environment can be traced leading to computational speedups even if the whole training cycle can't be fully traced. There may be ways in future to avoid this issue - get in touch if you have any ideas! 

|    Algorithm     |                                                         Reference                                                         |
|:----------------:|:-------------------------------------------------------------------------------------------------------------------------:|
| MPC (using iCEM) |                           [Paper](https://proceedings.mlr.press/v155/pinneri21a/pinneri21a.pdf)                           |
|       TIP        | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/b90cb10d4dae058dd167388e76168c1b-Paper-Conference.pdf) |
|       PETS       |      [Paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/3de568f8597b94bda53149c7d7f5958c-Paper.pdf)       |
|      PILCO       |                [Paper](https://aiweb.cs.washington.edu/research/projects/aiweb/media/papers/tmpZj4RyS.pdf)                |

Currently implemented environments:

|  Environment   |                                                         Reference                                                         |
|:--------------:|:-------------------------------------------------------------------------------------------------------------------------:|
|    Pendulum    | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/b90cb10d4dae058dd167388e76168c1b-Paper-Conference.pdf) |
| Pilco Cartpole |                [Paper](https://aiweb.cs.washington.edu/research/projects/aiweb/media/papers/tmpZj4RyS.pdf)                |
|  Wet Chicken   |      [Paper](https://arxiv.org/pdf/1907.04902)       |



## Basic Usage

At it's core most Model-Based methods have some dynamics model to learn the transition (and reward) dynamics and a planner (or policy) to generate actions. We mirror this in our code base. We house different types of dynamics models in the 'dynamics model' folder that can be arbitrarily swapped. Agents (or planners/policies) are housed within their own folder.

New algorithms can be added in the 'agent' folder with their own directory/module stating the algorithm name. This must match the file name. The file itself has an agent class that imports from 'agent_base.py' and follows the naming convention of '\<Algorithm Name\>Agent'.

We use [Ml Collections](https://github.com/google/ml_collections) for Configs as it allows easy use with [ABSL](https://github.com/abseil/abseil-py) flags when running on a cluster for hyperparameter sweeps. There is a general config file and each agent must have it's own config file.

This is a multi-file implementation as we think it enables easier comparisons than single-file implementations, but we try and keep most of the coniditional logic abstracted to the unique 'agent' files, rather than housing this type of logic in the shared running files!

## Installation

This is a Python package that should be installed into a virtual environment. Start by cloning this repo from Github:

git clone https://github.com/JamesRudd-Jones/JaxModel-BasedRL.git

The package can then be installed into a virtual environment by adding it as a local dependency. We recommend [PDM](https://pdm-project.org/en/latest/) or [Poetry](https://python-poetry.org/).

However, there are some current work-arounds we make as of now, but will have some more concrete solutions in the future. Please carry out the following:

Go on your '.venv' and to 'site-packages/plum' and adjust 'function.py:478' to log to debug not info.
For some reason GPJax has extensive logging and we are unsure how else to turn this off. 

Remove 'check_positive' if statement from GPJax 'parameters.py:143' since it disenables jit operations over a GP, it seems to be fine without it but maybe will lead to some errors?
Similarly, remove the if statement for  '_check_is_lower_triangular' on 'parameters.py:158'.

Further in GPJax 'dataset.py:92' and 'dataset.py:115' we have switched off '_check_shape' and '_check_precision' respectively as this prevented passing the GPJax Dataset through a vmap even if it wasn't being "vmapped".

## Contributing

We actively welcome contributions!

Please get in touch if you want to add an environment or algorithm, or have any questions regarding the implementations.
We also welcome any feedback regarding documentation!

For bug reports use Github's issues to track bugs, open a new issue and we will endeavour to get it fixed soon! 


## Future Roadmap

- Fully port over from the old [GPJax](https://github.com/aidanscannell/GPJax) to the new [GPJax](https://github.com/JaxGaussianProcesses/GPJax)
- Finish off the PILCO implementation
- Finsh off PETS implementation
- Add further dynamics models, such as Neural ODEs and equivalent.
- Incorporate a Model-Free wrapper for easy algorithm baseline comparison
- Add reward function learning alongisde dynamics models
- Create an easy environment wrapper for Generative Environments
