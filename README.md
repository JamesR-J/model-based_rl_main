# model-based_rl_main

Go on your .venv and to site-packages/plum and adjust function.py:478 to log to debug not info.
For some reason GPJax as crazy logging and cannot find how else to turn it off. There may be a better way so please let 
me know if so !! 

I have also removed check_positive if statement from Gpjax parameters.py:143 since it was disenabling jit operations 
over a GP, it seems to be fine without it but maybe will lead to some errors?

Also have done the same for _check_is_lower_triangular on parameters.py:158 for a similar reason