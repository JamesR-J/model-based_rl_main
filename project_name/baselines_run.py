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


def run_train(config):
    key = jrandom.key(config.SEED)

    # env = GymnaxPilcoCartPole()
    env = GymnaxPendulum()
    env_params = env.default_params
    action_dim = env.action_space(env_params).shape[0]  # TODO is there a better way to write this?

    # add plot functionality as required
    plot_fn = partial(plotters[config.ENV_NAME], env=env)  # TODO sort this out

    # normalise env if required
    if config.NORMALISE_ENV:
        env = NormalisedEnv(env, env_params)
        plot_fn = make_normalised_plot_fn(env, env_params, plot_fn)
    if config.GENERATIVE_ENV:
        env = GenerativeEnv(env, env_params)

    low = jnp.concatenate([env.observation_space(env_params).low, jnp.expand_dims(jnp.array(env.action_space(env_params).low), axis=0)])
    high = jnp.concatenate([env.observation_space(env_params).high, jnp.expand_dims(jnp.array(env.action_space(env_params).high), axis=0)])
    domain = [elt for elt in zip(low, high)]  # TODO something to refine this

    if config.GENERATIVE_ENV:
        if config.TELEPORT:
            mpc_func = utils.get_f_mpc_teleport
        else:
            mpc_func = utils.get_f_mpc
    else:
        raise NotImplementedError("If not generative env then we do not have a mpc_func yet")

    # set the initial obs, i.e. env.reset but able to set a consistent start point
    start_obs, start_env_state = utils.get_start_obs(env, env_params, key)

    # add the actor
    key, _key = jrandom.split(key)
    actor = utils.import_class_from_folder(config.AGENT_TYPE)(env=env, env_params=env_params, config=config, key=_key)

    # get some initial data for hyperparam tuning, unsure what else as of now? also a test set
    key, _key = jrandom.split(key)
    init_data_x, init_data_y = utils.get_initial_data(config, mpc_func, plot_fn, low, high, domain, env, env_params,
                                                      config.NUM_INIT_DATA, _key, train=True)
    key, _key = jrandom.split(key)
    test_data_x, test_data_y = utils.get_initial_data(config, mpc_func, plot_fn, low, high, domain, env, env_params,
                                                      config.NUM_INIT_DATA, _key)

    key, _key = jrandom.split(key)
    if config.PRETRAIN_HYPERPARAMS:
        pretrain_data_x, pretrain_data_y = utils.get_initial_data(config, mpc_func, plot_fn, low, high, domain, env,
                                                                  env_params,
                                                                  config.PRETRAIN_NUM_DATA, _key)
        key, _key = jrandom.split(key)
        train_state = actor.pretrain_params(init_data_x, init_data_y, pretrain_data_x, pretrain_data_y, _key)
    else:
        train_state = actor.create_train_state(init_data_x, init_data_y, _key)
        # TODO how does the above work if there is no data, can we use the start obs and a randoma action or nothing?

    @partial(jax.jit, static_argnums=(1,))
    def execute_gt_mpc(init_obs, f, key):
        key, _key = jrandom.split(key)
        full_path, test_mpc_data, all_returns = actor.run_algorithm_on_f(f, init_obs, train_state, key,
                                                                  horizon=env_params.horizon,
                                                                  actions_per_plan=actor.agent_config.ACTIONS_PER_PLAN)
        path_lengths = len(full_path[0])  # TODO should we turn the output into a dict for x and y ?
        true_path = actor.get_exe_path_crop(test_mpc_data[0], test_mpc_data[1])

        key, _key = jrandom.split(key)
        test_points = jax.tree_util.tree_map(lambda x: jrandom.choice(_key, x,
                                                                      (config.TEST_SET_SIZE // config.NUM_EVAL_TRIALS,)), true_path)
        # TODO ensure it samples the same pairs

        return true_path, test_points, path_lengths, all_returns

    # get some groundtruth data
    key, _key = jrandom.split(key)
    batch_key = jrandom.split(_key, config.NUM_EVAL_TRIALS)
    start_gt_time = time.time()
    true_paths, test_points, path_lengths, all_returns = jax.vmap(execute_gt_mpc, in_axes=(None, None, 0))(start_obs, mpc_func, batch_key)
    logging.info(f"Ground truth time taken = {time.time() - start_gt_time:.2f}s; "
                 f"Mean Return = {jnp.mean(all_returns):.2f}; Std Return = {jnp.std(all_returns):.2f}; "
                 f"Mean Path Lengths = {jnp.mean(path_lengths)}; ")

    # Plot groundtruth paths and print info
    ax_gt = None
    fig_gt = None
    for idx in range(true_paths["exe_path_x"].shape[0]):  # TODO sort the dodgy plotting can we add inside the vmap?
        plot_path = PlotTuple(x=true_paths["exe_path_x"][idx], y=true_paths["exe_path_y"][idx])
        ax_gt, fig_gt = plot_fn(plot_path, ax_gt, fig_gt, domain, "samp")
    if fig_gt and config.SAVE_FIGURES:
        fig_gt.suptitle("Ground Truth Eval")
        neatplot.save_figure("figures/gt", "png", fig=fig_gt)

    # this runs the main loop of learning
    def _main_loop(curr_obs_O, data_x, data_y, train_state, env_state, key):
        for i in range(config.NUM_ITERS):
            # log some info that we need basically
            logging.info("---" * 5 + f" Start iteration i={i} " + "---" * 5)
            logging.info(f"Length of data.x: {len(data_x)}; Length of data.y: {len(data_y)}")

            # TODO some if statement if our input data does not exist as not using generative approach, i.e. the first step

            # get next point
            x_next_OPA, exe_path, curr_obs_O, train_state, acq_val, key = actor.get_next_point(curr_obs_O, train_state, key)

            # periodically run evaluation and plot
            if i % config.EVAL_FREQ == 0 or i + 1 == config.NUM_ITERS:
                def _eval_trial(start_obs, start_env_state, key):
                    def _env_step(env_runner_state, unused):
                        obs_O, env_state, key = env_runner_state
                        key, _key = jrandom.split(key)
                        action_1A, _, _ = actor.execute_mpc(actor.make_postmean_func(), obs_O, train_state, _key, horizon=1, actions_per_plan=1)
                        action_A = jnp.squeeze(action_1A, axis=0)
                        key, _key = jrandom.split(key)
                        nobs_O, new_env_state, reward, done, info = env.step(_key, env_state, action_A, env_params)
                        return (nobs_O, new_env_state, key), (nobs_O, reward, action_A)

                    key, _key = jrandom.split(key)
                    _, (nobs_SO, real_rewards_S, real_actions_SA) = jax.lax.scan(_env_step, (start_obs, start_env_state, _key), None, env_params.horizon)
                    real_obs_SP1O = jnp.concatenate((jnp.expand_dims(start_obs, axis=0), nobs_SO))
                    real_returns_1 = actor._compute_returns(jnp.expand_dims(real_rewards_S, axis=0))  # TODO sort this out at some point
                    real_path_x_SOPA = jnp.concatenate((real_obs_SP1O[:-1], real_actions_SA), axis=-1)
                    real_path_y_SO = real_obs_SP1O[1:] - real_obs_SP1O[:-1]
                    key, _key = jrandom.split(key)
                    real_path_y_hat_SO = actor.make_postmean_func2()(real_path_x_SOPA, None, None, train_state, _key)
                    # TODO unsure how to fix the above
                    mse = 0.5 * jnp.mean(jnp.sum(jnp.square(real_path_y_SO - real_path_y_hat_SO), axis=1))

                    return (RealPath(x=real_path_x_SOPA, y=real_path_y_SO, y_hat=real_path_y_hat_SO),
                            jnp.squeeze(real_returns_1), jnp.mean(real_returns_1), jnp.std(real_returns_1), jnp.mean(mse))

                key, _key = jrandom.split(key)
                batch_key = jrandom.split(_key, config.NUM_EVAL_TRIALS)
                start_obs, start_env_state = utils.get_start_obs(env, env_params, key)
                real_paths_mpc, real_returns, mean_returns, std_returns, mse = jax.vmap(_eval_trial, in_axes=(None, None, 0))(start_obs, start_env_state, batch_key)
                logging.info(f"Eval Returns = {real_returns}; Mean = {jnp.mean(mean_returns):.2f}; "
                             f"Std = {jnp.std(std_returns):.2f}")  # TODO check the std
                logging.info(f"Model MSE = {jnp.mean(mse):.2f}")

                # TODO add testing on the random test_data that we created initially

                utils.make_plots(plot_fn, domain,
                                 PlotTuple(x=true_paths["exe_path_x"][-1], y=true_paths["exe_path_y"][-1]),
                                 PlotTuple(x=data_x, y=data_y),
                                 env, env_params, config, exe_path, real_paths_mpc, x_next_OPA, i)

            # Query function, update data
            key, _key = jrandom.split(key)
            if config.GENERATIVE_ENV:
                y_next_O = mpc_func(jnp.expand_dims(x_next_OPA, axis=0), env, env_params, train_state, _key)
                new_env_state = "UHOH"
                if config.ROLLOUT_SAMPLING:
                    delta = y_next_O[-env.obs_dim:]
                    nobs_O = actor._update_fn(curr_obs_O, delta, env, env_params)
                    # TODO sort the above out, it works when curr_obs doesn't change
                else:
                    raise NotImplementedError("When is it not rollout sampling?")
            else:
                action_A = x_next_OPA[-action_dim:]
                nobs_O, new_env_state, reward, done, info = env.step(_key, env_state, action_A, env_params)
                y_next_O = nobs_O - curr_obs_O
            # the above should match

            data_x = jnp.concatenate((data_x, jnp.expand_dims(x_next_OPA, axis=0)))
            data_y = jnp.concatenate((data_y, jnp.expand_dims(y_next_O, axis=0)))

            env_state = new_env_state
            curr_obs_O = nobs_O

            # TODO somehow update the dataset that is used in the GP but also generally in the loop although so dodgy
            # TODO how can we generalise this to the GP but also to the flax based approaches
            if config.AGENT_TYPE != "PETS":
                train_state["train_data_x"] = data_x
                train_state["train_data_y"] = data_y

    _main_loop(start_obs, init_data_x, init_data_y, train_state, start_env_state, key)

    return
