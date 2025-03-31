import jax
from typing import Tuple, Any
import chex
from functools import partial
import jax.numpy as jnp


class DynamicsModelBase:  # TODO sort this oot
    def __init__(self, env, env_params, config, agent_config, key):
        self.config = config
        self.env = env
        self.env_params = env_params
        self.agent_config = agent_config

        self.obs_dim = env.obs_dim
        self.action_dim = self.env.action_space(env_params).shape[0]
        self.input_dim = self.obs_dim + self.action_dim
        if config.LEARN_REWARD:
            self.output_dim = self.obs_dim + 1  # TODO is it just one?
        else:
            self.output_dim = self.obs_dim

    def create_train_state(self, init_data_x, init_data_y, key):
        raise NotImplementedError

    def pretrain_params(self, init_data_x, init_data_y, pretrain_data_x, pretrain_data_y, key):
        raise NotImplementedError

    def predict_y(self):
        raise NotImplementedError

    def predict_f(self):
        raise NotImplementedError

    def make_postmean_func(self):
        raise NotImplementedError

    def get_next_point(self, curr_obs, train_state, key):
        raise NotImplementedError