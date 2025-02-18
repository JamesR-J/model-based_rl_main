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
from project_name.dynamics_models import NeuralNetDynamicsModel


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

    env, env_params = gymnax.make("MountainCarContinuous-v0")
    # TODO make these envs in gymnax style for continuous control

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

    key, _key = jrandom.split(key)
    init_data_x, init_data_y = get_initial_data(config, env, f, _key, None)  # TODO add plot function

    key, _key = jrandom.split(key)
    test_data_x, test_data_y = get_initial_data(config, env, f, _key, None)  # TODO add plot function

    key, _key = jrandom.split(key)
    dynamics_model = NeuralNetDynamicsModel(init_obs, env.action_space().sample(_key), hidden_dims=[50, 50],
                                            hidden_activations=jax.nn.swish, is_probabilistic=True)

    ### Gets some groundtruth data, we batch it please at some points  # TODO batch this
    true_path, test_mpc_data = actor.agent.run_algorithm_on_f(f, key)

    def train():
        key = jax.random.PRNGKey(config.SEED)

        actor = Agent(env=env, env_params=env_params, config=config, utils=None, key=key)
        train_state, mem_state = actor.initialise()

        new_config = actor.agent.config  # TODO why this causing errors?

        reset_key = jrandom.split(key, config.NUM_ENVS)
        init_obs_NO, env_state = jax.vmap(env.reset, in_axes=(0, None), axis_name="batch_axis")(reset_key, env_params)

        runner_state = (train_state, mem_state, env_state, init_obs_NO, jnp.zeros(config.NUM_ENVS, dtype=bool), key)

        def _run_inner_update(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _run_episode_step(runner_state, unused):
                # take initial env_state
                train_state, mem_state, env_state, obs_NO, done_N, key = runner_state

                mem_state, action_NA, key = actor.act(train_state, mem_state, obs_NO, done_N, key)

                # step in env
                # key, _key = jrandom.split(key)
                key_step = jrandom.split(key, config.NUM_ENVS)
                nobs_NO, env_state, reward_N, ndone_N, info = jax.vmap(env.step, in_axes=(0, 0, 0, None),
                                                              axis_name="batch_axis")(key_step,
                                                                                      env_state,
                                                                                      action_NA,
                                                                                      env_params
                                                                                      )

                # mem_state = actor.update_encoding(train_state, mem_state, nobs_NO, action_NA, reward_N, ndone_N, key)

                transition = Transition(ndone_N, action_NA, reward_N, obs_NO, mem_state,
                                        # env_state,  # TODO have added for info purposes
                                        info)

                return (train_state, mem_state, env_state, nobs_NO, ndone_N, key), transition

            ((train_state, mem_state, env_state, last_obs_NO, last_done_N, key),
             trajectory_batch_LNZ) = jax.lax.scan(_run_episode_step, runner_state, None, actor.agent.agent_config.NUM_INNER_STEPS)
            # TODO have changed the above too

            train_state, mem_state, agent_info, key = actor.update(train_state, mem_state,  last_obs_NO,  last_done_N,
                                                                   key, trajectory_batch_LNZ)

            def callback(traj_batch, env_stats, agent_stats, update_steps):
                avg_episode_end_reward = traj_batch.info["reward"][traj_batch.info["returned_episode"]].mean()
                metric_dict = {"Total Steps": update_steps * config.NUM_ENVS * actor.agent.agent_config.NUM_INNER_STEPS,
                               "Total_Episodes": update_steps * config.NUM_ENVS,
                               # "avg_reward": traj_batch.reward.mean(),
                               "avg_returns": traj_batch.info["returned_episode_returns"][traj_batch.info["returned_episode"]].mean(),
                               "avg_episode_end_reward": jnp.where(jnp.isnan(avg_episode_end_reward), -100.0, avg_episode_end_reward),
                               "avg_episode_length": traj_batch.info["returned_episode_lengths"][traj_batch.info["returned_episode"]].mean(),
                               "avg_action": traj_batch.action.mean()}
                # TODO have changed the num_inner_steps above

                # shape is LN, so we are averaging over the num_envs and episode
                for item in agent_info:
                    metric_dict[f"{item}"] = agent_stats[item]

                # print(traj_batch.reward.shape)
                # print(traj_batch.reward.mean())
                # print(traj_batch.info["reward"][traj_batch.info["returned_episode"]])
                # print("NEW ONE")

                # print(traj_batch.info["reward"][traj_batch.info["returned_episode"]].mean())

                wandb.log(metric_dict)

            # env_stats = jax.tree_util.tree_map(lambda x: x.mean(), utils.visitation(env_state,
            #                                                                         collapsed_trajectory_batch,
            #                                                                         obs))

            jax.experimental.io_callback(callback, None, trajectory_batch_LNZ, env_state,
                                         agent_info, update_steps)

            update_steps = update_steps + 1

            return ((train_state, mem_state, env_state, last_obs_NO, last_done_N, key), update_steps), None

        runner_state, _ = jax.lax.scan(_run_inner_update,(runner_state, 0),None, new_config.NUM_EPISODES)
        # TODO have changed the above

        return {"runner_state": runner_state}

    return train


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=True):
        train = run_train(config)
        out = train()
