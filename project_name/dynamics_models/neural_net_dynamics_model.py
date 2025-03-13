from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
from project_name.dynamics_models import DynamicsModelBase
from ml_collections import ConfigDict


class NeuralNetDynamicsModel(DynamicsModelBase):
    def __init__(self, env, env_params, config, agent_config, key):
        super().__init__(env, env_params, config, agent_config, key)
        # Do not import its own config, should inherit from the agent config

        self.network = SimpleNetwork(agent_config, self.obs_dim, self.action_dim)

        self.tx = optax.adam(self.agent_config.LR)

    def create_train_state(self, init_data_x, init_data_y, key):
        def create_ensemble_state(key):
            params = self.network.init(key, jnp.zeros((1, self.obs_dim)), jnp.zeros((1, self.action_dim)))
            ind_state = TrainState.create(apply_fn=self.network.apply, params=params, tx=self.tx)
            return ind_state

        ensemble_keys = jrandom.split(key, self.agent_config.NUM_ENSEMBLE)
        return jax.vmap(create_ensemble_state, in_axes=(0,))(ensemble_keys)

    def predict(self, params: Dict, state: Dict, obs, action, key):
        raw_output = self._compute_net_output(params, state, obs, action)

        raw_prediction = reparameterized_gaussian_sampler(raw_output, rng_key)
        # raw_prediction = raw_output  # TODO add in deterministic output

        # raw_prediction = denormalize(state["normalizer"]["output"], raw_prediction)
        return self._next_obs_comp(obs, raw_prediction)

    def log_likelihood(self, params: Dict, state: Dict, obs, action, next_obs):
        """Computes the log-likelihood of the target induced by (obs, next_obs) with respect to the model,
        conditioned on (obs, action).

        Note: For deterministic models, the log-likelihood is computed as if the network output is the mean of a
        multivariate Gaussian with identity covariance.

        Args:
            params: Dictionary of model parameters.
            state: Dictionary of model state.
            obs: Environment observation.
            action: Action.
            next_obs: Next environment observation.

        Returns:
            Log-likelihood.
        """
        # raw_output = self._compute_net_output(params, state, obs, action)
        # targ = normalize(state["normalizer"]["output"], self._targ_comp(obs, next_obs))
        #
        # if self.is_probabilistic:
        #     gaussian_params = raw_output
        # else:
        #     gaussian_params = {"mean": raw_output, "stddev": 1.}
        #
        # return gaussian_log_prob(gaussian_params, targ)
        return

    def _compute_net_output(self, params, state, obs, action):
        # unnormalized_net_input = self._compute_unnormalized_net_input(obs, action)
        # return self._internal_net.forward(
        #     params["internal_net"],
        #     state["internal_net"],
        #     normalize(state["normalizer"]["input"], unnormalized_net_input)
        # )
        return

    def _compute_unnormalized_net_input(self, obs, action):
        preproc_obs = self._obs_preproc(obs)
        return jnp.concatenate([preproc_obs, action])


class SimpleNetwork(nn.Module):
    agent_config: ConfigDict
    obs_dim: int
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs, actions):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs = nn.Dense(self.agent_config.HIDDEN_SIZE - self.action_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs)
        x = jnp.concatenate((obs, actions), axis=-1)

        x = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(self.obs_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

        return x