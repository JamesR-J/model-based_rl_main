# Jax Model-Based RL 

no current benchamrk place for model-based rl

implementated algorithms as a table:
PETS
PILCO
cite papres aswell

## Basic Usage

## Installation

## Contributing

## Future Roadmap


Go on your .venv and to site-packages/plum and adjust function.py:478 to log to debug not info.
For some reason GPJax as crazy logging and cannot find how else to turn it off. There may be a better way so please let 
me know if so !! 

I have also removed check_positive if statement from Gpjax parameters.py:143 since it was disenabling jit operations 
over a GP, it seems to be fine without it but maybe will lead to some errors?

Also have done the same for _check_is_lower_triangular on parameters.py:158 for a similar reason

FURTHER in gpjax package dataset.py:92 and dataset.py:115 I have switched off _check_shape and _check_precision 
respectively as this prevented passing the gpjax Dataset through a vmap even if it wasn't being "vmapped"
