from absl import app
from project_name.baselines_run import run_train
import wandb
from project_name.config import get_config  # TODO dodge need to know how to fix this
import jax
from jax.lib import xla_bridge
import logging


jax.config.update("jax_enable_x64", True)  # TODO unsure if need or not but will check results


# TODO MUST DO a key check for the MPC

# TODO sort out this aidan scannall GPJax types, and also to make it work well

# env, env_params = gymnax.make("MountainCarContinuous-v0")
# TODO make these envs in gymnax style for continuous control


def main(_):
    config = get_config()

    # wandb.init(project="RL_BASE",
    #     entity=config.WANDB_ENTITY,
    #     config=config,
    #     # group="ks_tests",
    #     group="continuous_tests",
    #     mode=config.WANDB
    # )

    config.DEVICE = jax.extend.backend.get_backend().platform
    logging.info(f"Current JAX Device: {config.DEVICE}")

    with jax.disable_jit(disable=config.DISABLE_JIT):
        train = run_train(config)

    print("FINITO")


if __name__ == '__main__':
    app.run(main)
