import jax.numpy as jnp
import jax
import jax.random as jrandom
import wandb
from typing import NamedTuple
import chex
from project_name.agents import Agent
from project_name.envs.wrappers import NormalisedEnv, GenerativeEnv, make_normalised_plot_fn
from project_name.utils import Transition, EvalTransition, PlotTuple, RealPath
from project_name import utils
import sys
# from project_name import envs
import logging
from project_name.envs.gymnax_pilco_cartpole import GymnaxPilcoCartPole  # TODO add some register thing here instead
from project_name.envs.gymnax_pendulum import GymnaxPendulum  # TODO add some register thing here instead
from functools import partial
import time
from project_name.viz import plotters, plot
import neatplot
from jaxtyping import Float, install_import_hook
from jax.experimental import checkify
import bifurcagym

with install_import_hook("gpjax", "beartype.beartype"):
    import logging
    logging.getLogger('gpjax').setLevel(logging.WARNING)
    import gpjax


def run_train(config):
    key = jrandom.key(config.SEED)

    # env = GymnaxPilcoCartPole()
    # env = GymnaxPendulum()
    env = bifurcagym.make("Pendulum-v0",
                          cont_state=True,
                          cont_action=True,
                          normalised=config.NORMALISE_ENV,
                          autoreset=False)
    action_dim = env.action_space().shape[0]

    # add plot functionality as required
    plot_fn = partial(plotters[config.ENV_NAME], env=env)  # TODO sort this out

    # normalise env if required
    if config.NORMALISE_ENV:
        plot_fn = make_normalised_plot_fn(env, plot_fn)

    # TODO add normalised plotting somehow to the environments

    low = jnp.concatenate([env.observation_space().low, jnp.expand_dims(env.action_space().low, axis=0)])
    high = jnp.concatenate([env.observation_space().high, jnp.expand_dims(env.action_space().high, axis=0)])
    domain = jnp.concatenate((jnp.expand_dims(low, axis=-1), jnp.expand_dims(high, axis=-1)), axis=-1)

    if config.GENERATIVE_ENV:
        mpc_func = utils.generative_env_transition
    else:
        mpc_func = utils.env_transition
        # TODO add in the usual step as a function if possible, may need some major reworking
        # raise NotImplementedError("If not generative env then we do not have a mpc_func yet")

    # set the initial obs, i.e. env.reset
    # TODO should be able to set a consistent start point defined in config
    start_obs, start_env_state = utils.get_start_obs(env, key)

    # add the actor
    key, _key = jrandom.split(key)
    actor = utils.import_class_from_folder(config.AGENT_TYPE)(env=env, config=config, key=_key)

    # get some initial data for training/system identification. This should be agent specific
    # TODO is it possible to have this equal to zero when using certain setups? Or set this to the start_obs
    key, _key = jrandom.split(key)
    sys_id_data_x, syd_id_data_y = utils.get_initial_data(config, mpc_func, plot_fn, low, high, domain, env,
                                                      actor.agent_config.SYS_ID_DATA, _key, train=True)
    sys_id_dataset = gpjax.Dataset(sys_id_data_x, syd_id_data_y)
    # key, _key = jrandom.split(key)
    # test_data_x, test_data_y = utils.get_initial_data(config, mpc_func, plot_fn, low, high, domain, env, env_params,
    #                                                   config.NUM_INIT_DATA, _key)

    key, _key = jrandom.split(key)
    if config.PRETRAIN_HYPERPARAMS:
        pretrain_data_x, pretrain_data_y = utils.get_initial_data(config, mpc_func, plot_fn, low, high, domain, env,
                                                                  config.PRETRAIN_NUM_DATA, _key)
        pretrain_data = gpjax.Dataset(pretrain_data_x, pretrain_data_y)
        key, _key = jrandom.split(key)
        train_state = actor.pretrain_params(sys_id_dataset, pretrain_data, _key)
    else:
        train_state = actor.create_train_state(sys_id_dataset, _key)
        # TODO how does the above work if there is no data, can we use the start obs and a randoma action or nothing?

    # get some groundtruth data for evaluation
    if actor.agent_config.ROLLOUT_SAMPLING:
        key, _key = jrandom.split(key)
        batch_key = jrandom.split(_key, config.NUM_EVAL_TRIALS)
        start_gt_time = time.time()
        true_paths, test_points, all_returns = (jax.vmap(actor.execute_gt_mpc, in_axes=(None, None, None, None, None, 0))
                                                              (start_obs,
                                                               mpc_func,
                                                               start_env_state,
                                                               train_state,
                                                               (sys_id_dataset.X, sys_id_dataset.y),
                                                               # TODO ideally sort the above at some point
                                                               batch_key))
        logging.info(f"Ground truth time taken = {time.time() - start_gt_time:.2f}s; "
                     f"Mean Return = {jnp.mean(all_returns):.2f}; Std Return = {jnp.std(all_returns):.2f}")

        # Plot groundtruth paths and print info
        ax_gt = None
        fig_gt = None
        for idx in range(true_paths["exe_path_x"].shape[0]):  # TODO sort the dodgy plotting can we add inside the vmap?
            plot_path = PlotTuple(x=true_paths["exe_path_x"][idx], y=true_paths["exe_path_y"][idx])
            ax_gt, fig_gt = plot_fn(plot_path, ax_gt, fig_gt, domain, "samp")
        if fig_gt and config.SAVE_FIGURES:
            fig_gt.suptitle("Ground Truth Eval")
            neatplot.save_figure("figures/gt", "png", fig=fig_gt)
    else:
        true_paths = {"exe_path_x": jnp.zeros((1, 10, 3)),
                      "exe_path_y": jnp.zeros((1, 10, 2))}
        # TODO how can we give some groundtruth even if not using MPC?

    # this runs the main loop of learning
    def _main_loop(curr_obs_O, curr_env_state, train_data, train_state, key):
        global_returns = 0
        for step_idx in range(0, config.NUM_ITERS):
            # log some info that we need basically
            step_start_time = time.time()
            logging.info("---" * 5 + f" Start iteration i={step_idx} " + "---" * 5)
            logging.info(f"Length of data.x: {len(train_data.X)}; Length of data.y: {len(train_data.y)}")

            # TODO some if statement if our input data does not exist as not using generative approach, i.e. the first step

            # get next point
            x_next_OPA, exe_path, train_state, acq_val, key = actor.get_next_point(curr_obs_O,
                                                                                   curr_env_state,
                                                                                   train_state,
                                                                                   train_data,
                                                                                   step_idx,
                                                                                   key)
            assert jnp.allclose(curr_obs_O, x_next_OPA[:env.observation_space().shape[0]]), "For rollout cases, we can only give queries which are from the current state"

            # periodically run evaluation and plot
            if (step_idx % config.EVAL_FREQ == 0 or step_idx + 1 == config.NUM_ITERS):
                key, _key = jrandom.split(key)
                batch_key = jrandom.split(_key, config.NUM_EVAL_TRIALS)
                start_obs, start_env_state = utils.get_start_obs(env, key)
                real_paths_mpc, real_returns, mean_returns, std_returns, mse = jax.vmap(actor.evaluate, in_axes=(None, None, None, None, 0))(start_obs, start_env_state, train_state, (train_data.X, train_data.y), batch_key)
                logging.info(f"Eval Returns = {real_returns}; Mean = {jnp.mean(mean_returns):.2f}; "
                             f"Std = {jnp.std(std_returns):.2f}")  # TODO check the std
                logging.info(f"Model MSE = {jnp.mean(mse):.2f}")

                # TODO add testing on the random test_data that we created initially

                utils.make_plots(plot_fn, domain,
                                 PlotTuple(x=true_paths["exe_path_x"][-1], y=true_paths["exe_path_y"][-1]),
                                 PlotTuple(x=train_data.X, y=train_data.y),
                                 env, config, actor.agent_config, exe_path, real_paths_mpc, x_next_OPA,
                                 step_idx)

            # Query function, update data
            key, _key = jrandom.split(key)
            action_A = x_next_OPA[-action_dim:]
            nobs_O, y_next_O, next_env_state = mpc_func(curr_obs_O, action_A, env, curr_env_state, train_state, train_data, _key)
            key, _key = jrandom.split(key)
            reward = env.reward_function(action_A, curr_env_state, next_env_state, _key)
            global_returns += reward

            logging.info(f"Iteration i={step_idx} Action: {action_A}")
            logging.info(f"Iteration i={step_idx} State : {nobs_O}")
            logging.info(f"iteration i={step_idx} Return so far: {global_returns}")

            train_data = train_data + gpjax.Dataset(X=jnp.expand_dims(x_next_OPA, axis=0), y=jnp.expand_dims(y_next_O, axis=0))
            # TODO will the above work with PETS as well?
            curr_obs_O = nobs_O
            curr_env_state = next_env_state
            logging.info(f"Iteration time taken - {time.time() - step_start_time:.1f}s")

    _main_loop(start_obs, start_env_state, sys_id_dataset, train_state, key)

    return
