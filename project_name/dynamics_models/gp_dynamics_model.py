import jax
import jax.numpy as jnp
import jax.random as jrandom
import gpjax
from flax import struct
from flax.training import train_state
import optax
from typing import List, Tuple, Dict, Optional, NamedTuple, Any
from functools import partial
import GPJax_AScannell as gpjaxas
import optax
import gpjax
from project_name.dynamics_models import DynamicsModelBase


class MOGP(DynamicsModelBase):
    def __init__(self, env, env_params, config, agent_config, key):
        super().__init__(env, env_params, config, agent_config, key)
        num_latent_gps = self.obs_dim

        # TODO ls below is for cartpole
        # ls = jnp.array([[80430.37, 4.40, 116218.45, 108521.27, 103427.47], [290.01, 318.22, 0.39, 1.57, 33.17],
        #      [1063051.24, 1135236.37, 1239430.67, 25.09, 1176016.11], [331.70, 373.98, 0.32, 1.88, 39.83]], dtype=jnp.float64)
        ls = jnp.array([[2.27, 7.73, 138.94], [0.84, 288.15, 11.05]],
                       dtype=jnp.float64)
        alpha = jnp.array([0.26, 2.32, 11.59, 3.01], dtype=jnp.float64)  # TODO what is alpha? is this periodic?
        sigma = 0.01

        self.kernel = gpjaxas.kernels.SeparateIndependent([gpjaxas.kernels.SquaredExponential(lengthscales=ls[idx], variance=sigma) for idx in range(self.obs_dim)])
        self.likelihood = gpjaxas.likelihoods.Gaussian(variance=3.0)
        self.mean_function = gpjaxas.mean_functions.Zero(output_dim=self.action_dim)
        self.gp = gpjaxas.models.GPR(self.kernel, self.likelihood, self.mean_function, num_latent_gps=num_latent_gps)

    def create_train_state(self, init_data_x, init_data_y, key):
        params = self.gp.get_params()
        params["train_data_x"] = init_data_x
        params["train_data_y"] = init_data_y
        return params

    def get_post_mu_cov(self, XNew, params, full_cov=False):  # TODO if no data then return the prior mu and var
        mu, std = self.gp.predict_f(params, XNew, full_cov=full_cov)
        return mu, std

    def get_post_mu_cov_samples(self, XNew, params, key, full_cov=False):
        samples = self.gp.predict_f_samples(params, key, XNew, num_samples=1, full_cov=full_cov)
        return samples[0] # TODO works on the first sample as only one required for now
