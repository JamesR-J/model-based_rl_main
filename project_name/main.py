from absl import app
from project_name.baselines_run import run_train
import wandb
from project_name.config import get_config  # TODO dodge need to know how to fix this
import jax
from jax.lib import xla_bridge
import logging


jax.config.update("jax_enable_x64", True)  # TODO unsure if need or not but will check results

# TODO add in some hyperparm fit from initial data, OR some preloaded hyperparams

# TODO if we want to pretend we have no existing data how would this work?

# TODO sort out all the right shapes, so we don't have to arbitrarily squeeze and add dims in MPC, also can we avoid postmean_func2

# TODO replace the updating dataset with flashbax maybe?

# TODO sort out this aidan scannall GPJax types, and also to make it work well

# env, env_params = gymnax.make("MountainCarContinuous-v0")
# TODO make these envs in gymnax style for continuous control


"""
To check all this
curr_obs is the current obs of the environment
x_next is these obs appended with some action
nobs comes from passing x_next into the simulator
y_next = nobs - curr_obs
update_obs_fn literally just adds curr_obs + y_next to get the true nobs, and adds some teleportation if needed
"""


def main(_):
    config = get_config()

    # wandb.init(project="RL_BASE",
    #     entity=config.WANDB_ENTITY,
    #     config=config,
    #     # group="ks_tests",
    #     group="continuous_tests",
    #     mode=config.WANDB
    # )

    config.DEVICE = jax.lib.xla_bridge.get_backend().platform
    logging.info(f"Current JAX Device: {config.DEVICE}")

    with jax.disable_jit(disable=config.DISABLE_JIT):
        train = run_train(config)

    print("FINITO")


if __name__ == '__main__':
    app.run(main)
