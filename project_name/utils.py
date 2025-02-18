import sys
import os
import importlib
import jax.numpy as jnp
from typing import NamedTuple, Any, Mapping
import chex
import jax
from flax.training.train_state import TrainState
import flax


class MemoryState(NamedTuple):
    hstate: jnp.ndarray
    extras: Mapping[str, jnp.ndarray]


class TrainStateExt(TrainState):
    target_params: flax.core.FrozenDict


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    mem_state: MemoryState
    # env_state: Any  # TODO added this but can change
    info: jnp.ndarray


class TransitionFlashbax(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray

class EvalTransition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    distribution: Any
    spec_key: chex.PRNGKey
    env_state: jnp.ndarray


def flip_and_switch(tracer):
    return jnp.swapaxes(tracer, 0, 1)


def import_class_from_folder(folder_name):
    """
    Imports a class from a folder with the same name

    Args:
        folder_name (str): The name of the folder and potential class.

    Returns:
        The imported class, or None if import fails.
    """

    if not isinstance(folder_name, str):
        raise TypeError("folder_name must be a string.")

    # Check for multiple potential entries
    potential_path = os.path.join(os.curdir, "project_name", "agents",
                                  folder_name)  # TODO the project_name addition ain't great

    if os.path.isdir(potential_path) and os.path.exists(
            os.path.join(potential_path, f"{folder_name}.py")):
        # Use importlib to dynamically import the module
        module_spec = importlib.util.spec_from_file_location(folder_name,
                                                             os.path.join(potential_path, f"{folder_name}.py"))
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        # Retrieve the class from the imported module
        return getattr(module, f"{folder_name}Agent")

    else:
        print(f"Error: Folder '{folder_name}' not found in any search paths.")
        return None
