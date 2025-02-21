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
from project_name.dynamics_models import NeuralNetDynamicsModel, MultiGpfsGp, BatchMultiGpfsGp
from project_name.envs.gymnax_pilco_cartpole import GymnaxPilcoCartPole  # TODO add some register thing here instead


@struct.dataclass  # TODO dodgy for now and need to change
class EnvState(environment.EnvState):
    position: jnp.ndarray
    velocity: jnp.ndarray
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

    # TODO add the plot functionality as required

    # if config.NORMALISE_ENV:  # TODO add this for gymnax etc
    #     env = NormalizedEnv(env)
    #     if reward_function is not None:
    #         reward_function = make_normalized_reward_function(env, reward_function)
    #     # plot_fn = make_normalized_plot_fn(env, plot_fn)  # TODO add plot normalisation aswell

    def get_f_mpc(env, use_info_delta=False):
        obs_dim = len(env.observation_space(env_params).low)

        def f(x_OPA, key):
            obs_O = x_OPA[:obs_dim]
            action_A = x_OPA[obs_dim:]
            env_state = EnvState(position=obs_O[0], velocity=obs_O[1], time=0)  # TODO specific for mountaincar, need to generalise this
            nobs_O, _, _, _, info = env.step(key, env_state, action_A, env_params)
            if use_info_delta:
                return info["delta_obs"]
            else:
                return nobs_O - obs_O

        return jax.vmap(f)

    f = get_f_mpc(env)

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
        data_y_LO = f(data_x_LOPA, batch_key)

        # # Plot initial data  # TODO add data plotting
        # ax_obs_init, fig_obs_init = plot_fn(path=None, domain=domain)
        # if ax_obs_init is not None and config.save_figures:
        #     plot(ax_obs_init, data.x, "o", color="k", ms=1)
        #     fig_obs_init.suptitle("Initial Observations")
        #     neatplot.save_figure(str(dumper.expdir / "obs_init"), "png", fig=fig_obs_init)
        return data_x_LOPA, data_y_LO

    # key, _key = jrandom.split(key)
    # init_data_x, init_data_y = get_initial_data(config, env, f, _key, None)  # TODO add plot function
    #
    # key, _key = jrandom.split(key)
    # test_data_x, test_data_y = get_initial_data(config, env, f, _key, None)  # TODO add plot function

    key, _key = jrandom.split(key)
    dynamics_model = NeuralNetDynamicsModel(init_obs, env.action_space().sample(_key), hidden_dims=[50, 50],
                                            hidden_activations=jax.nn.swish, is_probabilistic=True)

    dynamics_model = MultiGpfsGp()

    ### Gets some groundtruth data, we batch it please at some points  # TODO batch this
    def execute_gt_mpc(init_obs, key):
        key, _key = jrandom.split(key)
        full_path, test_mpc_data = actor.agent.run_algorithm_on_f(None, init_obs, _key)  # TODO need to use the dervied f and not the one inside MPC
        path_lengths = len(full_path[0])  # TODO should we turn the output into a dict for x and y ?
        true_path = actor.agent.get_exe_path_crop(full_path)

        key, _key = jrandom.split(key)
        test_points = jax.tree_util.tree_map(lambda x: jrandom.choice(_key, x,
                                                                      (config.TEST_SET_SIZE // config.NUM_EVAL_TRIALS, )), true_path)
        # TODO ensure it samples the same pairs

        return true_path, test_points

    batch_key = jrandom.split(key, config.NUM_EVAL_TRIALS)
    true_path, test_points = jax.vmap(execute_gt_mpc, in_axes=(None, 0))(init_obs, batch_key)
    flattened_true_path = jax.tree_util.tree_map(lambda x: x.reshape((x[0] * x[1], -1)), true_path)
    flattened_test_points = jax.tree_util.tree_map(lambda x: x.reshape((config.TEST_SET_SIZE, -1)), test_points)

    # # Plot groundtruth paths and print info  # TODO plot these gt
    # ax_gt, fig_gt = plot_fn(true_path, ax_gt, fig_gt, domain, "samp")
    # returns.append(compute_return(output[2], 1))
    # stats = {"Mean Return": np.mean(returns), "Std Return:": np.std(returns)}

    # TODO add some hyperparameter fit on the GT data or if we are evaluatning the hyperparams

    def get_start_obs(obs=None):
        return obs

    init_obs = get_start_obs()

    def get_next_point(init_data, curr_obs):
        # TODO some if satatement if our input data does not exist as not using generative approach

        # TODO if statement if using an acquisition function, idk how to do this so that we don't require if statement

        # If using MPC then
        model = dynamics_model(init_data)

        def _postmean_fn(x):
            mu_list, std_list = model.get_post_mu_cov(x, full_cov=False)
            mu_list = np.array(mu_list)
            mu_tup_for_x = list(zip(*mu_list))
            return mu_tup_for_x

        policy = partial(actor.agent.execute_mpc, f=_postmean_fn(model))
        action = policy(curr_obs)
        x_next = jnp.concatenate([curr_obs, action])



        return

    def _main_loop():
        # log some info that we need basically

        # get next point

        # periodically run evaluation and plot

        # Query function, update data

        return

    _main_loop()

    return train


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=True):
        train = run_train(config)
        out = train()
