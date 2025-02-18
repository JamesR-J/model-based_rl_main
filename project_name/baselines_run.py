import jax.numpy as jnp
import jax
import jax.random as jrandom
from project_name.config import get_config  # TODO dodge need to know how to fix this
import wandb
from typing import NamedTuple
import chex
from agents import Agent
from project_name.envs.wrappers import NormalizedEnv, make_normalized_reward_function
from utils import Transition, EvalTransition
import sys
import gymnasium as gym
from project_name import envs
import logging


def run_train(config):
    key, _key = jrandom.split(config.SEED)

    env = gym.make(config.ENV_NAME)
    reward_function = envs.reward_functions[config.env.name]

    # TODO add the plot functionality as required

    if config.NORMALISE_ENV:
        env = NormalizedEnv(env)
        if reward_function is not None:
            reward_function = make_normalized_reward_function(env, reward_function, config.alg.gd_opt)
        # TODO add plot normalisation aswell

    # set the initial obs, i.e. env.reset
    init_obs, _ = env.reset()
    logging.info(f"Start obs: {init_obs}")

    actor = Agent(env=env, env_params=None, config=config, utils=None, key=_key)

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
