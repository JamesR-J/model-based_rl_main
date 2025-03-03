import jax.numpy as jnp
import jax
import jax.random as jrandom
from project_name.config import get_config  # TODO dodge need to know how to fix this
import wandb
from typing import NamedTuple
import chex
from project_name.agents import Agent
from project_name.envs.wrappers import NormalizedEnv, make_normalized_reward_function
from project_name.util.control_util import get_f_batch_mpc
from project_name.utils import Transition, EvalTransition
import sys
import gymnasium as gym
from project_name import envs
import logging
import numpy as np
import gymnax
from gymnax.environments import environment
from flax import struct
from project_name import dynamics_models
from project_name.envs.gymnax_pilco_cartpole import GymnaxPilcoCartPole  # TODO add some register thing here instead
import gpjax
from functools import partial


@struct.dataclass  # TODO dodgy for now and need to change the gymnax envs to be better for this
class EnvState(environment.EnvState):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int


def run_train(config):
    key = jrandom.PRNGKey(config.SEED)
    key, _key = jrandom.split(key)

    env = gym.make(config.ENV_NAME)
    reward_function = envs.reward_functions[config.ENV_NAME]

    # env, env_params = gymnax.make("MountainCarContinuous-v0")
    # TODO make these envs in gymnax style for continuous control

    env = GymnaxPilcoCartPole()
    env_params = env.default_params
    obs_dim = len(env.observation_space(env_params).low)  # TODO is there a better way to write this?
    action_dim = env.action_space().shape[0]  # TODO same for this

    # TODO add the plot functionality as required

    # if config.NORMALISE_ENV:  # TODO add this for gymnax etc
    #     env = NormalizedEnv(env)
    #     if reward_function is not None:
    #         reward_function = make_normalized_reward_function(env, reward_function)
    #     # plot_fn = make_normalized_plot_fn(env, plot_fn)  # TODO add plot normalisation aswell

    def _get_f_mpc(x_OPA, key, use_info_delta=False):  # TODO this should be generalised out of class at some point
        obs_O = x_OPA[:obs_dim]
        action_A = x_OPA[obs_dim:]
        env_state = EnvState(x=obs_O[0], x_dot=obs_O[1], theta=obs_O[2], theta_dot=obs_O[3], time=0)  # TODO specific for cartpole, need to generalise this
        nobs_O, _, _, _, info = env.step(key, env_state, action_A, env_params)
        return nobs_O - obs_O

    f = _get_f_mpc  # TODO kinda weak but okay for now
    batch_f = jax.vmap(_get_f_mpc)

    # set the initial obs, i.e. env.reset
    key, _key = jrandom.split(key)
    init_obs, init_env_state = env.reset(_key)
    logging.info(f"Start obs: {init_obs}")

    actor = Agent(env=env, env_params=env_params, config=config, utils=None, key=_key)

    # TODO update below so we can work in envs which are not generative
    def get_initial_data(config, env, f, key, plot_fn):
        def unif_random_sample_domain(low, high, key, n=1):
            unscaled_random_sample = jrandom.uniform(key, shape=(n, low.shape[0]))
            scaled_random_sample = low[..., :] + (high[..., :] - low[..., :]) * unscaled_random_sample
            return scaled_random_sample

        low = jnp.concatenate((env.observation_space(env_params).low, jnp.expand_dims(jnp.array(env.action_space().low,), axis=0)))
        high = jnp.concatenate((env.observation_space(env_params).high, jnp.expand_dims(jnp.array(env.action_space().high,), axis=0)))
        # TODO is there a better way to do the above

        data_x_LOPA = unif_random_sample_domain(low, high, key, n=config.NUM_INIT_DATA)
        batch_key = jrandom.split(key, config.NUM_INIT_DATA)
        data_y_LO = batch_f(data_x_LOPA, batch_key)

        # # Plot initial data  # TODO add data plotting
        # ax_obs_init, fig_obs_init = plot_fn(path=None, domain=domain)
        # if ax_obs_init is not None and config.save_figures:
        #     plot(ax_obs_init, data.x, "o", color="k", ms=1)
        #     fig_obs_init.suptitle("Initial Observations")
        #     neatplot.save_figure(str(dumper.expdir / "obs_init"), "png", fig=fig_obs_init)
        return data_x_LOPA, data_y_LO

    key, _key = jrandom.split(key)
    init_data_x, init_data_y = get_initial_data(config, env, f, _key, None)  # TODO add plot function

    # key, _key = jrandom.split(key)
    # test_data_x, test_data_y = get_initial_data(config, env, f, _key, None)  # TODO add plot function

    key, _key = jrandom.split(key)
    # dynamics_model = dynamics_model.NeuralNetDynamicsModel(init_obs, env.action_space().sample(_key), hidden_dims=[50, 50],
    #                                         hidden_activations=jax.nn.swish, is_probabilistic=True)
    dynamics_model = dynamics_models.MOGP(env, env_params, config, None, key)
    dynamics_model_train_state = dynamics_model.create_train_state(init_obs)  # TODO kinda dodge but adds the first obs to the GP dataset as otherwise have an empty params that would not work I think?
    # TODO as mentioned above we add obs and some arbitrary action, but this may impact the model greatly so should fix this
    # TODO generalise this and add some more dynamics models

    ### Gets some groundtruth data, we batch it please at some points  # TODO batch this
    def execute_gt_mpc(init_obs, key):
        key, _key = jrandom.split(key)
        full_path, test_mpc_data = actor.agent.run_algorithm_on_f(f, init_obs, _key,
                                                                  horizon=config.ENV_HORIZON,  # TODO p sure it is this, but should come from the env and not from config?
                                                                  actions_per_plan=actor.agent.agent_config.ACTIONS_PER_PLAN)
        path_lengths = len(full_path[0])  # TODO should we turn the output into a dict for x and y ?
        true_path = actor.agent.get_exe_path_crop(full_path)

        key, _key = jrandom.split(key)
        test_points = jax.tree_util.tree_map(lambda x: jrandom.choice(_key, x,
                                                                      (config.TEST_SET_SIZE // config.NUM_EVAL_TRIALS, )), true_path)
        # TODO ensure it samples the same pairs

        return true_path, test_points

    batch_key = jrandom.split(key, config.NUM_EVAL_TRIALS)
    true_path, test_points = jax.vmap(execute_gt_mpc, in_axes=(None, 0))(init_obs, batch_key)
    flattened_true_path = jax.tree_util.tree_map(lambda x: x.reshape((x.shape[0] * x.shape[1], -1)), true_path)
    flattened_test_points = jax.tree_util.tree_map(lambda x: x.reshape((config.TEST_SET_SIZE, -1)), test_points)

    # # Plot groundtruth paths and print info  # TODO plot these gt
    # ax_gt, fig_gt = plot_fn(true_path, ax_gt, fig_gt, domain, "samp")
    # returns.append(compute_return(output[2], 1))
    # stats = {"Mean Return": np.mean(returns), "Std Return:": np.std(returns)}

    # TODO add some hyperparameter fit on the GT data or if we are evaluating the hyperparams

    def get_start_obs(key, obs=None):  # TODO some if statement if have some fixed start point
        key, _key = jrandom.split(key)
        obs, env_state = env.reset(_key)
        return obs, env_state

    curr_obs, curr_env_state = get_start_obs(key)

    def get_next_point(curr_obs, init_data_x, init_data_y, key):
        # TODO some if statement if our input data does not exist as not using generative approach, i.e. the first step

        # TODO if statement if using an acquisition function, idk how to do this so that we don't require if statement
        # idea here is to run a batch of MPC on different functions from the posterior
        # so can we sample like a batch of params? so that we can just call the GP on these params in a VMAPPED setting

        # we need to make a posterior params list that is vmappable for future use
        key, _key = jrandom.split(key)
        samples = dynamics_model.return_posterior_samples(jnp.concatenate((jnp.expand_dims(curr_obs, axis=0), jnp.zeros((1, 1))), axis=-1),
                                                          dynamics_model_train_state, _key)



        # If using MPC then
        def make_postmean_func(model):
            def _postmean_fn(x, key):
                x = jnp.expand_dims(x, axis=0)  # TODO adding in a number of data points dim as this required for GPJax
                mu, std = model.get_post_mu_cov(x, dynamics_model_train_state, full_cov=False)
                return jnp.squeeze(mu, axis=0)  # TODO this removes the first num_data dim
            return _postmean_fn

        key, _key = jrandom.split(key)
        action, exe_path = actor.agent.execute_mpc(make_postmean_func(dynamics_model), curr_obs, key)
        x_next = jnp.concatenate((jnp.expand_dims(curr_obs, axis=0), action), axis=-1)

        return x_next, exe_path

    def _main_loop(curr_obs_O, init_data_x, init_data_y, env_state, f, key):  # TODO run this for config.num_iters
        # log some info that we need basically

        # get next point
        x_next_BOPA, exe_path = get_next_point(curr_obs_O, init_data_x, init_data_y, key)
        x_next_OPA = jnp.squeeze(x_next_BOPA, axis=0)

        # periodically run evaluation and plot

        # Query function, update data
        key, _key = jrandom.split(key)
        if config.GENERATIVE_ENV:
            y_next = f(x_next_OPA, _key)
            nobs_O = y_next + curr_obs_O
        else:
            action_A = x_next_OPA[-action_dim:]
            nobs_O, new_env_state, reward, done, info = env.step(_key, env_state, action_A, env_params)
            y_next = nobs_O - curr_obs_O

        return

    _main_loop(curr_obs, init_data_x, init_data_y, curr_env_state, f, key)

    return train


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=True):
        train = run_train(config)
        out = train()
