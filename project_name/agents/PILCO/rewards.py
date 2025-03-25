import jax.numpy as jnp
import jax.random as jrandom
from ml_collections import ConfigDict
from typing import Any, Callable, Sequence
import jax
import flax.linen as nn


class ExponentialReward(nn.Module):
    obs_dim: int
    w_init: Callable = nn.initializers.uniform()
    t_init: Callable = nn.initializers.uniform()

    @nn.compact
    def __call__(self, m, s):
        """
        Reward function, calculating mean and variance of rewards, given
        mean and variance of state distribution, along with the target State
        and a weight matrix.
        Input m : [1, k]
        Input s : [k, k]

        Output M : [1, 1]
        Output S  : [1, 1]
        """
        W = self.param("W", self.w_init, (self.obs_dim, self.obs_dim))
        t = self.param("t", self.t_init, (1, self.obs_dim))

        SW = s @ W

        iSpW = jnp.transpose(jnp.linalg.solve((jnp.eye(self.obs_dim) + SW), jnp.transpose(W)))

        muR = (jnp.exp(-(m - t) @ iSpW @ jnp.transpose(m - t) / 2) /
               jnp.sqrt(jnp.linalg.det(jnp.eye(self.obs_dim) + SW)))

        i2SpW = jnp.transpose(jnp.linalg.solve((jnp.eye(self.obs_dim) + 2 * SW), jnp.transpose(W)))

        r2 = jnp.exp(-(m - t) @ i2SpW @ jnp.transpose(m - t)) / jnp.sqrt(jnp.linalg.det(jnp.eye(self.obs_dim) + 2 * SW))

        sR = r2 - muR @ muR
        return muR, sR


# class LinearReward(objax.Module):
#     def __init__(self, state_dim, W):
#         self.state_dim = state_dim
#         self.W = objax.StateVar(jnp.reshape(W, (state_dim, 1)))
#
#     def compute_reward(self, m, s):
#         muR = jnp.reshape(m, (1, self.state_dim)) @ self.W
#         sR = jnp.transpose(self.W) @ s @ self.W
#         return muR, sR
#
#
# class CombinedRewards(objax.Module):
#     def __init__(self, state_dim, rewards=[], coefs=None):
#         self.state_dim = state_dim
#         self.base_rewards = rewards
#         if coefs is not None:
#             self.coefs = objax.StateVar(coefs)
#         else:
#             self.coefs = objax.StateVar(jnp.ones(len(rewards)))
#
#     def compute_reward(self, m, s):
#         total_output_mean = 0
#         total_output_covariance = 0
#         for reward, coef in zip(self.base_rewards, self.coefs):
#             output_mean, output_covariance = reward.compute_reward(m, s)
#             total_output_mean += coef * output_mean
#             total_output_covariance += coef**2 * output_covariance
#
#         return total_output_mean, total_output_covariance
