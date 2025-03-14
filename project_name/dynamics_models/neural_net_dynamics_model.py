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

    def predict(self, x, train_state, key):
        obs_BO = x[..., :self.obs_dim]
        actions_BA = x[..., self.obs_dim:]
        y_pred = train_state.apply_fn(train_state.params, obs_BO, actions_BA)

        # TODO add in some deterministic check thingo as the below is stochastic and the above deterministic

        min_stddev = 1e-5
        max_stddev = 100
        delta = max_stddev - min_stddev
        out_dim = y_pred.shape[-1] // 2
        stddev_logit = y_pred[..., out_dim:]
        std = min_stddev + delta * jax.nn.sigmoid(4 * (stddev_logit / delta))
        mean = y_pred[..., :out_dim]
        logstd = jnp.log(std)

        key, _key = jrandom.split(key)  # TODO do I need this split?
        standard_normal_sample = jrandom.normal(_key, shape=mean.shape)

        return mean + std * standard_normal_sample, std

    def log_likelihood(self, x, nobs_BO, train_state, key):
        """Computes the log-likelihood of the target induced by (obs, next_obs) with respect to the model,
        conditioned on (obs, action).

        Note: For deterministic models, the log-likelihood is computed as if the network output is the mean of a
        multivariate Gaussian with identity covariance.
        """

        obs_BO = x[..., :self.obs_dim]
        actions_BA = x[..., self.obs_dim:]
        y_pred = train_state.apply_fn(train_state.params, obs_BO, actions_BA)

        # TODO add in some deterministic check thingo as the below is stochastic and the above deterministic

        min_stddev = 1e-5
        max_stddev = 100
        delta = max_stddev - min_stddev
        out_dim = y_pred.shape[-1] // 2
        stddev_logit = y_pred[..., out_dim:]
        std = min_stddev + delta * jax.nn.sigmoid(4 * (stddev_logit / delta))
        mean = y_pred[..., :out_dim]
        logstd = jnp.log(std)

        targ = nobs_BO - obs_BO  # TODO check this as in the original it is nobs - obs

        # if self.is_probabilistic:
        #     gaussian_params = raw_output
        # else:
        #     gaussian_params = {"mean": raw_output, "stddev": 1.}

        dim = mean.size
        weighted_mse = 0.5 * jnp.sum(jnp.square((targ - mean) / std))
        log_det_cov = jnp.sum(logstd)
        return -(weighted_mse + log_det_cov + (dim / 2) * jnp.log(2 * jnp.pi))

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