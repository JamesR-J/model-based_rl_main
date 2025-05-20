import jax
from typing import Tuple, Any
import chex
from functools import partial
import jax.numpy as jnp


class AgentBase:  # TODO sort this oot
    def __init__(self, env, config, key):
        self.config = config
        self.env = env

    def create_train_state(self, init_data, key):
        raise NotImplementedError

    def pretrain_params(self, init_data, pretrain_data, key):
        raise NotImplementedError

    # def reset_memory(self, mem_state) -> Any:
    #     raise NotImplementedError
    #
    def act(self, train_state: Any, mem_state: Any, ac_in: chex.Array, key: chex.PRNGKey) -> Tuple[Any, chex.Array, chex.Array, chex.Array, chex.PRNGKey]:
        raise NotImplementedError
    #
    # @partial(jax.jit, static_argnums=(0,))
    # def update(self, runner_state: Any, agent: int, traj_batch: chex.Array, all_mem_state) -> Tuple[Any, Any, Any, Any, chex.PRNGKey]:
    #     train_state, mem_state, env_state, ac_in, key = runner_state
    #     return train_state, mem_state, env_state, None, key
    #
    # @partial(jax.jit, static_argnums=(0,))
    # def update_encoding(self, train_state: Any, mem_state: Any, agent: int, obs_batch: chex.Array, action: chex.Array,
    #                     reward: chex.Array, done: chex.Array, key: chex.PRNGKey) -> Any:
    #     return mem_state

    def make_postmean_func(self):
        raise NotImplementedError

    def get_next_point(self, curr_obs, train_state, train_data, step_idx, key):
        raise NotImplementedError

    def evaluate(self, start_obs, start_env_state, train_state, train_data, key):
        raise NotImplementedError