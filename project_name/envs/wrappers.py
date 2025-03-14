# from gymnasium import Env, spaces
from gymnax.environments import environment, spaces
from copy import deepcopy
import numpy as np
from project_name.utils import PlotTuple
import jax.numpy as jnp
import jax.random as jrandom
from project_name import utils
import jax
from functools import partial
import chex
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union, overload


jax.config.update("jax_enable_x64", True)


TEnvState = TypeVar("TEnvState", bound="EnvState")
TEnvParams = TypeVar("TEnvParams", bound="EnvParams")


class NormalisedEnv(environment.Environment):
    def __init__(self, wrapped_env, env_params):
        """
        Normalises obs to be between -1 and 1
        """
        self._wrapped_env = wrapped_env
        self.unnorm_action_space = self._wrapped_env.action_space()
        self.unnorm_observation_space = self._wrapped_env.observation_space(env_params)
        self.unnorm_obs_space_size = self.unnorm_observation_space.high - self.unnorm_observation_space.low
        self.unnorm_action_space_size = self.unnorm_action_space.high - self.unnorm_action_space.low

    def action_space(self, params=None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-1, 1, shape=(self.unnorm_action_space.shape[0],))

    def observation_space(self, params=None) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(low=-np.ones_like(self.unnorm_observation_space.low,),
                          high=np.ones_like(self.unnorm_observation_space.high,),
                          shape=(self.unnorm_observation_space.shape[0],))

    @property
    def wrapped_env(self):
        return self._wrapped_env

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: TEnvParams) -> Tuple[chex.Array, TEnvState]:
        unnorm_obs, env_state = self._wrapped_env.reset(key, params)
        return self.normalise_obs(unnorm_obs), env_state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: chex.PRNGKey, state: TEnvState, action: Union[int, float, chex.Array],
             params: TEnvParams) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs, new_env_state, rew, done, info = self._wrapped_env.step_env(key, state, unnorm_action, params)

        unnorm_delta_obs = info["delta_obs"]
        norm_delta_obs = unnorm_delta_obs / self.unnorm_obs_space_size * 2
        info["delta_obs"] = norm_delta_obs

        return self.normalise_obs(unnorm_obs), new_env_state, rew, done, info

    @partial(jax.jit, static_argnums=(0,))
    def generative_step_env(self, key, norm_obs, action, params):
        unnorm_init_obs = self.unnormalise_obs(norm_obs)
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs, new_env_state, rew, done, info = self._wrapped_env.generative_step_env(key, unnorm_init_obs, unnorm_action, params)

        unnorm_delta_obs = info["delta_obs"]
        norm_delta_obs = unnorm_delta_obs / self.unnorm_obs_space_size * 2
        info["delta_obs"] = norm_delta_obs

        return self.normalise_obs(unnorm_obs), new_env_state, rew, done, info

    def reward_function(self, x, next_obs, params):
        norm_obs = x[..., :self._wrapped_env.obs_dim]
        action = x[..., self._wrapped_env.obs_dim:]
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs = self.unnormalise_obs(norm_obs)
        unnorm_x = jnp.concatenate([unnorm_obs, unnorm_action], axis=-1)
        unnorm_y = self.unnormalise_obs(next_obs)
        rewards = self._wrapped_env.reward_function(unnorm_x, unnorm_y, params)

        return rewards

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_env)

    def normalise_obs(self, obs):
        low = self.unnorm_observation_space.low
        size = self.unnorm_obs_space_size
        pos_obs = obs - low
        norm_obs = (pos_obs / size * 2) - 1
        return norm_obs

    def unnormalise_obs(self, obs):
        low = self.unnorm_observation_space.low
        size = self.unnorm_obs_space_size
        obs01 = (obs + 1) / 2
        obs_ranged = obs01 * size
        unnorm_obs = obs_ranged + low
        return unnorm_obs

    def unnormalise_action(self, action):
        low = self.unnorm_action_space.low
        size = self.unnorm_action_space_size
        act01 = (action + 1) / 2
        act_ranged = act01 * size
        unnorm_act = act_ranged + low
        return unnorm_act

    def normalise_action(self, action):
        low = self.unnorm_action_space.low
        size = self.unnorm_action_space_size
        pos_action = action - low
        norm_action = (pos_action / size * 2) - 1
        return norm_action


class GenerativeEnv(object):
    def __init__(self, wrapper_env, env_params):
        self._wrapper_env = wrapper_env
        self.env_params = env_params

    @property
    def wrapper_env(self):
        return self._wrapper_env

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self, key: chex.PRNGKey, orig_obs, action: Union[int, float, chex.Array],
                        params: Optional[TEnvParams] = None) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self._wrapper_env.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self._wrapper_env.generative_step_env(key, orig_obs, action, params)
        obs_re, state_re = self._wrapper_env.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    def __getattr__(self, attr):
        if attr == "_wrapper_env":
            raise AttributeError()
        return getattr(self._wrapper_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapper_env)



def make_normalised_plot_fn(norm_env, env_params, plot_fn):
    obs_dim = norm_env.observation_space().low.size
    wrapped_env = norm_env.wrapped_env
    # Set domain
    low = np.concatenate([wrapped_env.observation_space(env_params).low,
                          np.expand_dims(np.array(wrapped_env.action_space().low), axis=0)])
    high = np.concatenate([wrapped_env.observation_space(env_params).high,
                           np.expand_dims(np.array(wrapped_env.action_space().high), axis=0)])
    unnorm_domain = [elt for elt in zip(low, high)]

    def norm_plot_fn(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
        if path:
            x = np.array(path.x)
            norm_obs = x[..., :obs_dim]
            action = x[..., obs_dim:]
            unnorm_action = norm_env.unnormalise_action(action)
            unnorm_obs = norm_env.unnormalise_obs(norm_obs)
            unnorm_x = np.concatenate([unnorm_obs, unnorm_action], axis=-1)
            try:
                y = path.y
                unnorm_y = norm_env.unnormalise_obs(y)
            except AttributeError:
                pass
            path = PlotTuple(x=unnorm_x, y=unnorm_y)
        return plot_fn(path, ax=ax, fig=fig, domain=unnorm_domain, path_str=path_str, env=env)

    return norm_plot_fn


def test_obs(wrapped_env, obs):
    unnorm_obs = wrapped_env.unnormalise_obs(obs)
    renorm_obs = wrapped_env.normalise_obs(unnorm_obs)
    assert np.allclose(obs, renorm_obs), f"Original obs {obs} not close to renormalised obs {renorm_obs}"


def test_rew_fn(gt_rew, old_obs, action, obs, wrapped_env, env_params):
    x = jnp.concatenate([old_obs, action])
    y = obs
    norm_rew = wrapped_env.reward_function(x, y, env_params)
    assert jnp.allclose(gt_rew, norm_rew), f"gt_rew: {gt_rew}, norm_rew: {norm_rew}"


def test_update_function(start_obs, action, delta_obs, next_obs, update_fn, env, env_params):
    x = jnp.concatenate([start_obs, action], axis=-1)
    updated_next_obs = update_fn(x, delta_obs, env, env_params)
    assert jnp.allclose(next_obs, updated_next_obs), f"Next obs: {next_obs} and updated next obs: {updated_next_obs}"


def test():
    import sys

    sys.path.append(".")
    from gymnax_pendulum import GymnaxPendulum
    from gymnax_pilco_cartpole import GymnaxPilcoCartPole

    sys.path.append("..")
    key = jrandom.PRNGKey(42)
    env = GymnaxPendulum()
    # env = GymnaxPilcoCartPole()
    env_params = env.default_params
    wrapped_env = NormalisedEnv(env, env_params)
    regular_update_fn = utils.update_obs_fn
    # wrapped_reward = make_normalised_reward_function(wrapped_env, pendulum_reward)
    teleport_update_fn = utils.update_obs_fn_teleport
    key, _key = jrandom.split(key)
    obs, env_state = wrapped_env.reset(_key)
    test_obs(wrapped_env, obs)
    done = False
    total_rew = 0
    observations = []
    next_observations = []
    rewards = []
    actions = []
    teleport_deltas = []
    for _ in range(env_params.horizon):
        old_obs = obs
        observations.append(old_obs)
        key, _key = jrandom.split(key)
        action = wrapped_env.action_space().sample(_key)
        actions.append(action)
        key, _key = jrandom.split(key)
        obs, env_state, rew, done, info = wrapped_env.step(_key, env_state, action, env_params)
        next_observations.append(obs)
        total_rew += rew
        standard_delta_obs = obs - old_obs
        teleport_deltas.append(info["delta_obs"])
        test_update_function(old_obs, action, standard_delta_obs, obs, regular_update_fn, wrapped_env, env_params)
        test_update_function(old_obs, action, info["delta_obs"], obs, teleport_update_fn, wrapped_env, env_params)
        rewards.append(rew)
        test_obs(wrapped_env, obs)
        test_rew_fn(rew, old_obs, action, obs, wrapped_env, env_params)
        if done:
            break
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    teleport_deltas = np.array(teleport_deltas)
    x = jnp.concatenate([observations, actions], axis=1)
    teleport_next_obs = teleport_update_fn(x, teleport_deltas, wrapped_env, env_params)
    assert np.allclose(teleport_next_obs, next_observations)
    test_rewards = wrapped_env.reward_function(x, next_observations, env_params)
    assert np.allclose(rewards, test_rewards), f"Rewards: {rewards} not equal to test rewards: {test_rewards}"
    print(f"passed!, rew={total_rew}")


if __name__ == "__main__":
    test()
