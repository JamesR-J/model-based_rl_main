import sys
import os
import importlib
import jax.numpy as jnp
from typing import NamedTuple, Any, Mapping
import chex
import jax
from flax.training.train_state import TrainState
import flax
from project_name.viz import make_plot_obs, scatter, plot
import neatplot

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


class MPCTransition(NamedTuple):
    obs: jnp.ndarray
    action:jnp.ndarray
    reward:jnp.ndarray

class MPCTransitionXY(NamedTuple):
    obs: jnp.ndarray
    action:jnp.ndarray
    reward:jnp.ndarray
    x: jnp.ndarray
    y: jnp.ndarray

class MPCTransitionXYR(NamedTuple):
    obs: jnp.ndarray
    action:jnp.ndarray
    reward:jnp.ndarray
    x: jnp.ndarray
    y: jnp.ndarray
    returns: jnp.ndarray

class PlotTuple(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray

class RealPath(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    y_hat: jnp.ndarray



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


def make_plots(plot_fn, domain, true_path, data, env, env_params, config, exe_path_list, real_paths_mpc, x_next, i):
    if len(data.x) == 0:
        return
    # Initialize various axes and figures
    ax_all, fig_all = plot_fn(path=None, domain=domain)
    ax_postmean, fig_postmean = plot_fn(path=None, domain=domain)
    ax_samp, fig_samp = plot_fn(path=None, domain=domain)
    ax_obs, fig_obs = plot_fn(path=None, domain=domain)
    # Plot true path and posterior path samples
    if true_path is not None:
        ax_all, fig_all = plot_fn(true_path, ax_all, fig_all, domain, "true")
    if ax_all is None:
        return
    # Plot observations
    x_obs = make_plot_obs(data.x, env, env_params, config.NORMALISE_ENV)
    scatter(ax_all, x_obs, color="grey", s=10, alpha=0.3)
    plot(ax_obs, x_obs, "o", color="k", ms=1)

    # Plot execution path posterior samples
    for idx in range(exe_path_list["exe_path_x"].shape[0]):  # TODO sort the dodgy plotting
        path = PlotTuple(x=exe_path_list["exe_path_x"][idx], y=exe_path_list["exe_path_y"][idx])
        ax_all, fig_all = plot_fn(path, ax_all, fig_all, domain, "samp")
        ax_samp, fig_samp = plot_fn(path, ax_samp, fig_samp, domain, "samp")

    # plot posterior mean paths
    for idx in range(real_paths_mpc.x.shape[0]):
        path = RealPath(x=real_paths_mpc.x[idx], y=real_paths_mpc.y[idx], y_hat=real_paths_mpc.y_hat[idx])
        ax_all, fig_all = plot_fn(path, ax_all, fig_all, domain, "postmean")
        ax_postmean, fig_postmean = plot_fn(path, ax_postmean, fig_postmean, domain, "samp")

    # Plot x_next
    x = make_plot_obs(x_next, env, env_params, config.NORMALISE_ENV)
    scatter(ax_all, x, facecolors="deeppink", edgecolors="k", s=120, zorder=100)
    plot(ax_obs, x, "o", mfc="deeppink", mec="k", ms=12, zorder=100)

    try:
        # set titles if there is a single axes
        ax_all.set_title(f"All - Iteration {i}")
        ax_postmean.set_title(f"Posterior Mean Eval - Iteration {i}")
        ax_samp.set_title(f"Posterior Samples - Iteration {i}")
        ax_obs.set_title(f"Observations - Iteration {i}")
    except AttributeError:
        # set titles for figures if they are multi-axes
        fig_all.suptitle(f"All - Iteration {i}")
        fig_postmean.suptitle(f"Posterior Mean Eval - Iteration {i}")
        fig_samp.suptitle(f"Posterior Samples - Iteration {i}")
        fig_obs.suptitle(f"Observations - Iteration {i}")

    if config.SAVE_FIGURES:
        # Save figure at end of evaluation
        neatplot.save_figure(f"figures/all_{i}", "png", fig=fig_all)
        neatplot.save_figure(f"figures/postmean_{i}", "png", fig=fig_postmean)
        neatplot.save_figure(f"figures/samp_{i}", "png", fig=fig_samp)
        neatplot.save_figure(f"figures/obs_{i}", "png", fig=fig_obs)
