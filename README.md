# Jax Model-Based RL 

This is the beginnings of a benchmark suite of Model-Based Reinforcement Learning approaches based in Jax. 

### Why Jax?

Much of the 

Issues are the changing dataset size at each real iteration aren't the most amenable to Jax tracers, but nonetheless
it still provides speedups.

no current benchamrk place for model-based rl

Currently implemented algorithms:

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

have some standard and shared dynamics models
new algoriths can be added in the agents with a folder 
add the specific naming style as it is essential

## Installation

## Contributing

## Future Roadmap

incorporate model-free wrapper to enable easy comparison
reward function learning
easy environment wrapper 



Go on your .venv and to site-packages/plum and adjust function.py:478 to log to debug not info.
For some reason GPJax as crazy logging and cannot find how else to turn it off. There may be a better way so please let 
me know if so !! 

I have also removed check_positive if statement from Gpjax parameters.py:143 since it was disenabling jit operations 
over a GP, it seems to be fine without it but maybe will lead to some errors?

Also have done the same for _check_is_lower_triangular on parameters.py:158 for a similar reason

FURTHER in gpjax package dataset.py:92 and dataset.py:115 I have switched off _check_shape and _check_precision 
respectively as this prevented passing the gpjax Dataset through a vmap even if it wasn't being "vmapped"
