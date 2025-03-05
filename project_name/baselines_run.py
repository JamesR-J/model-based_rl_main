import jax.numpy as jnp
import jax
import jax.random as jrandom
from project_name.config import get_config  # TODO dodge need to know how to fix this
import wandb
from typing import NamedTuple
import chex
from project_name.agents import Agent
from project_name.envs.wrappers import NormalizedEnv, make_normalized_reward_function
from project_name.utils import Transition, EvalTransition, PlotTuple, RealPath, make_plots
import sys
import gymnasium as gym
from project_name import envs
import logging
import gymnax
from gymnax.environments import environment
from flax import struct
from project_name import dynamics_models
from project_name.envs.gymnax_pilco_cartpole import GymnaxPilcoCartPole  # TODO add some register thing here instead
from project_name.envs.gymnax_pendulum import GymnaxPendulum  # TODO add some register thing here instead
from functools import partial
import time
from project_name.viz import plotters, plot
import numpy as np
import neatplot
from project_name.envs.wrappers import NormalizedEnv, make_normalized_reward_function, make_update_obs_fn, make_normalized_plot_fn
import pandas as pd


@struct.dataclass  # TODO dodgy for now and need to change the gymnax envs to be better for this
class EnvState(environment.EnvState):
    # x: jnp.ndarray
    # x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int


def run_train(config):
    key = jrandom.PRNGKey(config.SEED)

    # init the environment and reward function specific to it, unsure if still need this reward function thing?
    # env = gym.make(config.ENV_NAME)
    # reward_function = envs.reward_functions[config.ENV_NAME]  # TODO add the custom reward function thing

    # env = GymnaxPilcoCartPole()
    env = GymnaxPendulum()
    env_params = env.default_params
    obs_dim = len(env.observation_space(env_params).low)  # TODO is there a better way to write this?
    action_dim = env.action_space().shape[0]  # TODO same for this

    # add plot functionality as required
    plot_fn = partial(plotters[config.ENV_NAME], env=env)  # TODO sort this out

    # normalise env if required as well as reward function
    if config.NORMALISE_ENV:  # TODO this does not work so need to fix
        env = NormalizedEnv(env, env_params)
        # if reward_function is not None:
        #     reward_function = make_normalized_reward_function(env, reward_function)
        plot_fn = make_normalized_plot_fn(env, env_params, plot_fn)

    low = jnp.concatenate([env.observation_space(env_params).low, jnp.expand_dims(jnp.array(env.action_space().low), axis=0)])
    high = jnp.concatenate([env.observation_space(env_params).high, jnp.expand_dims(jnp.array(env.action_space().high), axis=0)])
    domain = [elt for elt in zip(low, high)]  # TODO something to refine this

    # TODO move the below somewhere better
    @partial(jax.jit, static_argnums=(2,))
    def _get_f_mpc(x_OPA, key, use_info_delta=False):  # TODO this should be generalised out of class at some point
        obs_O = x_OPA[:obs_dim]
        action_A = x_OPA[obs_dim:]
        # obs_O = env.normalize_obs(obs_O)  # TODO this dodgy fix still
        # env_state = EnvState(x=obs_O[0], x_dot=obs_O[1], theta=obs_O[2], theta_dot=obs_O[3], time=0)  # TODO specific for cartpole, need to generalise this
        env_state = EnvState(theta=obs_O[0], theta_dot=obs_O[1], time=0)  # TODO specific for pendulum, need to generalise
        nobs_O, _, _, _, info = env.step(key, env_state, action_A, env_params)
        return nobs_O - obs_O

    # set the initial obs, i.e. env.reset
    def get_start_obs(key, obs=None):  # TODO some if statement if have some fixed start point
        key, _key = jrandom.split(key)
        obs, env_state = env.reset(_key)
        logging.info(f"Start obs: {obs}")
        return obs, env_state
    start_obs, start_env_state = get_start_obs(key)

    # add the planner/actor
    key, _key = jrandom.split(key)
    actor = Agent(env=env, env_params=env_params, config=config, utils=None, key=_key)

    # add the dynamics model # TODO perhaps make this like part of a specific actor rather than separate here?
    key, _key = jrandom.split(key)
    # dynamics_model = dynamics_model.NeuralNetDynamicsModel(init_obs, env.action_space().sample(_key), hidden_dims=[50, 50],
    #                                         hidden_activations=jax.nn.swish, is_probabilistic=True)
    dynamics_model = dynamics_models.MOGP(env, env_params, config, None, key)
    # TODO generalise this and add some more dynamics models

    # @partial(jax.jit, static_argnums=(0, 1))
    def get_initial_data(f, plot_fn, key):
        def unif_random_sample_domain(low, high, key, n=1):
            unscaled_random_sample = jrandom.uniform(key, shape=(n, low.shape[0]))
            scaled_random_sample = low + (high - low) * unscaled_random_sample
            return scaled_random_sample

        data_x_LOPA = unif_random_sample_domain(low, high, key, n=config.NUM_INIT_DATA)
        if config.GENERATIVE_ENV:
            batch_key = jrandom.split(key, config.NUM_INIT_DATA)
            data_y_LO = jax.vmap(f)(data_x_LOPA, batch_key)
        else:
            raise NotImplementedError("If not generative env then we have to output nothing, unsure how to do in Jax")

        # Plot initial data
        ax_obs_init, fig_obs_init = plot_fn(path=None, domain=domain)
        if ax_obs_init is not None and config.SAVE_FIGURES:
            plot(ax_obs_init, data_x_LOPA, "o", color="k", ms=1)
            fig_obs_init.suptitle("Initial Observations")
            neatplot.save_figure("figures/obs_init", "png", fig=fig_obs_init)

        return data_x_LOPA, data_y_LO

    # get some initial data for something, unsure what as of now? also a test set
    key, _key = jrandom.split(key)
    init_data_x, init_data_y = get_initial_data(_get_f_mpc, plot_fn, _key)
    # key, _key = jrandom.split(key)
    # test_data_x, test_data_y = get_initial_data(jax.vmap(_get_f_mpc), plot_fn, _key

    @partial(jax.jit, static_argnums=(1,))
    def execute_gt_mpc(init_obs, f, key):
        key, _key = jrandom.split(key)
        full_path, test_mpc_data, all_returns = actor.agent.run_algorithm_on_f(f, init_obs, key,
                                                                  horizon=env_params.horizon,
                                                                  actions_per_plan=actor.agent.agent_config.ACTIONS_PER_PLAN)
        path_lengths = len(full_path[0])  # TODO should we turn the output into a dict for x and y ?
        true_path = actor.agent.get_exe_path_crop(test_mpc_data[0], test_mpc_data[1])

        key, _key = jrandom.split(key)
        test_points = jax.tree_util.tree_map(lambda x: jrandom.choice(_key, x,
                                                                      (config.TEST_SET_SIZE // config.NUM_EVAL_TRIALS, )), true_path)
        # TODO ensure it samples the same pairs

        return true_path, test_points, path_lengths, all_returns
    # TODO batching does nothing as the env does the same thing, why is this?

    # get some groundtruth data
    batch_key = jrandom.split(key, config.NUM_EVAL_TRIALS)
    start_gt_time = time.time()
    true_paths, test_points, path_lengths, all_returns = jax.vmap(execute_gt_mpc, in_axes=(None, None, 0))(start_obs, _get_f_mpc, batch_key)
    flattened_true_paths = jax.tree_util.tree_map(lambda x: x.reshape((x.shape[0] * x.shape[1], -1)), true_paths)
    # df_1 = pd.DataFrame(true_path["exe_path_x"][0])
    # df_2 = pd.DataFrame(true_path["exe_path_x"][1])
    # df_1.to_csv("data_file_1.csv")
    # df_2.to_csv("data_file_2.csv")
    # flattened_test_points = jax.tree_util.tree_map(lambda x: x.reshape((config.TEST_SET_SIZE, -1)), test_points)
    logging.info(f"Ground truth time taken = {time.time() - start_gt_time:.2f}s; "
                 f"Mean Return = {jnp.mean(all_returns):.2f}; Std Return = {jnp.std(all_returns):.2f}; "
                 f"Mean Path Lengths = {jnp.mean(path_lengths)}; ")

    # Plot groundtruth paths and print info
    ax_gt = None
    fig_gt = None
    for idx in range(true_paths["exe_path_x"].shape[0]):  # TODO sort the dodgy plotting
        plot_path = PlotTuple(x=true_paths["exe_path_x"][idx], y=true_paths["exe_path_y"][idx])
        ax_gt, fig_gt = plot_fn(plot_path, ax_gt, fig_gt, domain, "samp")
    if fig_gt and config.SAVE_FIGURES:
        fig_gt.suptitle("Ground Truth Eval")
        neatplot.save_figure("figures/gt", "png", fig=fig_gt)

    # TODO add some hyperparameter fit on the GT data or if we are evaluating the hyperparams

    def _main_loop(start_obs_O, env_state, key):  # TODO run this for config.num_iters
        key, _key = jrandom.split(key)
        init_data_x = jnp.expand_dims(jnp.concatenate((start_obs_O, env.action_space().sample(_key)), axis=-1), axis=0)
        init_data_y = jnp.zeros((1, obs_dim))

        dynamics_model_train_state = dynamics_model.create_train_state(init_data_x)
        # TODO kinda dodge but adds the first obs to the GP dataset as otherwise have an empty params that would not work I think?
        # TODO as mentioned above we add obs and some arbitrary action, but this may impact the model greatly so should fix this

        def create_range_of_train_state():
            return

        for i in range(config.NUM_ITERS):
            # log some info that we need basically
            logging.info("---" * 5 + f" Start iteration i={i} " + "---" * 5)
            logging.info(f"Length of data.x: {len(init_data_x)}")
            logging.info(f"Length of data.y: {len(init_data_y)}")

            def get_next_point(curr_obs, init_data_x, init_data_y, key):
                def make_postmean_func(model):
                    def _postmean_fn(x, key):
                        x = jnp.expand_dims(x, axis=0)  # TODO adding in num_data_points dim as required for GPJax
                        mu, std = model.get_post_mu_cov(x, dynamics_model_train_state, full_cov=False)
                        return jnp.squeeze(mu, axis=0)  # TODO this removes the first num_data_points dim

                    return _postmean_fn

                # TODO some if statement if our input data does not exist as not using generative approach, i.e. the first step

                # TODO if statement if using an acquisition function, idk how to do this so that we don't require if statement
                # idea here is to run a batch of MPC on different posterior functions, can we sample a batch of params? so that we can just call the GP on these params in a VMAPPED setting

                # we need to make a posterior params list that is vmappable for future use
                key, _key = jrandom.split(key)
                batch_key = jrandom.split(_key, actor.agent.agent_config.ACQUISITION_SAMPLES)
                _, exe_path_BSOPA = jax.vmap(actor.agent.execute_mpc, in_axes=(None, None, 0, None, None))(make_postmean_func(dynamics_model),
                                                                                         curr_obs,
                                                                                         batch_key,
                                                                                         env_params.horizon,
                                                                                         actor.agent.agent_config.ACTIONS_PER_PLAN)
                # TODO maybe some postmean function that takes specific keys dependant on the vamp batch

                # add in some test values
                key, _key = jrandom.split(key)
                x_test = jnp.concatenate((curr_obs, env.action_space().sample(_key)))

                #  now optimise the dynamics model with the x_test
                # take the exe_path_list that has been found with different posterior samples using iCEM
                # x_data and y_data are what ever you have currently
                # x_list is creating a trajectory from the iCEM samples using the posterior samples, how is this diff to exe path?
                key, _key = jrandom.split(key)
                x_next, acq_val = actor.agent.optimise(dynamics_model, dynamics_model_train_state, make_postmean_func(dynamics_model), exe_path_BSOPA, x_test, _key)
                # TODO figure out how we can batch this dynamics model etc that would be the easiest


                # samples = dynamics_model.return_posterior_samples(
                #     jnp.concatenate((jnp.expand_dims(curr_obs, axis=0), jnp.zeros((1, 1))), axis=-1),
                #     dynamics_model_train_state, _key)

                # If using MPC then
                # TODO some conditional for MPC here and for acquisition stuff above
                # key, _key = jrandom.split(key)
                # action, exe_path = actor.agent.execute_mpc(make_postmean_func(dynamics_model), curr_obs, key, horizon=1, actions_per_pan=1)
                # x_next = jnp.concatenate((jnp.expand_dims(curr_obs, axis=0), action), axis=-1)
                # x_next = jnp.squeeze(x_next, axis=0)

                return x_next, exe_path_BSOPA

            # get next point
            x_next_OPA, exe_path = get_next_point(start_obs_O, init_data_x, init_data_y, key)

            # periodically run evaluation and plot
            if i % config.EVAL_FREQ == 0 or i + 1 == config.NUM_ITERS:
                def make_postmean_func(model):
                    def _postmean_fn(x, key):
                        x = jnp.expand_dims(x, axis=0)  # TODO adding in num_data_points dim as required for GPJax
                        mu, std = model.get_post_mu_cov(x, dynamics_model_train_state, full_cov=False)
                        return jnp.squeeze(mu, axis=0)  # TODO this removes the first num_data_points dim

                    return _postmean_fn

                def make_postmean_func_2(model):  # TODO this is shite so sort it out
                    def _postmean_fn(x, key):
                        mu, std = model.get_post_mu_cov(x, dynamics_model_train_state, full_cov=False)
                        return mu

                    return _postmean_fn

                def _eval_trial(start_obs, start_env_state, key):
                    def _env_step(env_runner_state, unused):
                        obs_O, env_state, key = env_runner_state
                        key, _key = jrandom.split(key)
                        action_1A, _ = actor.agent.execute_mpc(make_postmean_func(dynamics_model), obs_O, _key,
                                                               horizon=1, actions_per_plan=1)
                        action_A = jnp.squeeze(action_1A, axis=0)
                        key, _key = jrandom.split(key)
                        nobs_O, new_env_state, reward, done, info = env.step(_key, env_state, action_A, env_params)

                        return (nobs_O, new_env_state, key), (nobs_O, reward, action_A)

                    key, _key = jrandom.split(key)
                    _, (nobs_SO, real_rewards_S, real_actions_SA) = jax.lax.scan(_env_step, (start_obs, start_env_state, _key), None, env_params.horizon)
                    real_obs_SP1O = jnp.concatenate((jnp.expand_dims(start_obs, axis=0), nobs_SO))
                    real_returns_1 = actor.agent._compute_returns(jnp.expand_dims(real_rewards_S, axis=0))  # TODO sort this out at some point
                    real_path_x_SOPA = jnp.concatenate((real_obs_SP1O[:-1], real_actions_SA), axis=-1)
                    real_path_y_SO = real_obs_SP1O[1:] - real_obs_SP1O[:-1]
                    key, _key = jrandom.split(key)
                    real_path_y_hat_SO = make_postmean_func_2(dynamics_model)(real_path_x_SOPA, _key)
                    mse = 0.5 * jnp.mean(jnp.sum(jnp.square(real_path_y_SO - real_path_y_hat_SO), axis=1))

                    return (RealPath(x=real_path_x_SOPA, y=real_path_y_SO, y_hat=real_path_y_hat_SO),
                            jnp.squeeze(real_returns_1), jnp.mean(real_returns_1), jnp.std(real_returns_1), jnp.mean(mse))

                batch_key = jrandom.split(key, config.NUM_EVAL_TRIALS)
                key, _key = jrandom.split(key)
                start_obs, start_env_state = get_start_obs(key)
                real_paths_mpc, real_returns, mean_returns, std_returns, mse = jax.vmap(_eval_trial, in_axes=(None, None, 0))(start_obs, start_env_state, batch_key)
                logging.info(f"Eval Returns = {real_returns}; Mean = {jnp.mean(mean_returns):.2f}; "
                             f"Std = {jnp.std(std_returns):.2f}")
                logging.info(f"Model MSE = {jnp.mean(mse):.2f}")

                # TODO add testing on the random test_data that we created initially

                make_plots(plot_fn,
                           domain,
                           PlotTuple(x=true_paths["exe_path_x"][-1], y=true_paths["exe_path_y"][-1]),
                           PlotTuple(x=init_data_x, y=init_data_y),
                           env,
                           env_params,
                           config,
                           exe_path,
                           real_paths_mpc,
                           x_next_OPA,
                           i)

            # Query function, update data
            key, _key = jrandom.split(key)
            if config.GENERATIVE_ENV & config.NO_ENV_RUN:
                y_next = _get_f_mpc(x_next_OPA, _key)
                nobs_O = y_next + start_obs_O
                new_env_state = "UHOH"
            else:
                action_A = x_next_OPA[-action_dim:]
                nobs_O, new_env_state, reward, done, info = env.step(_key, env_state, action_A, env_params)
                y_next = nobs_O - start_obs_O

            new_x = jnp.concatenate((init_data_x, jnp.expand_dims(x_next_OPA, axis=0)))
            new_y = jnp.concatenate((init_data_y, jnp.expand_dims(y_next, axis=0)))

            start_obs_O = nobs_O
            env_state = new_env_state
            init_data_x = new_x
            init_data_y = new_y

            # TODO somehow update the dataset that is used in the GP but also generally in the loop although so dodgy
            dynamics_model_train_state["train_data"] = init_data_x
            dynamics_model_train_state["q_mu"] = jnp.zeros_like(init_data_y)  # TODO again dodgy and should sort

    _main_loop(start_obs, start_env_state, key)

    return


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=True):
        train = run_train(config)
