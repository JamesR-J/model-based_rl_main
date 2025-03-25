import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrandom
from ml_collections import ConfigDict
from typing import Any, Callable, Sequence
import jax

class LinearController(nn.Module):
    obs_dim: int
    action_dim: int
    max_action: float
    w_init: Callable = nn.initializers.uniform()
    b_init: Callable = nn.initializers.uniform()

    @nn.compact
    def __call__(self, m, s):
        """
       Simple affine action:  M <- W(m-t) - b
       Only capable of one step predictions
       IN: mean (m) and variance (s) of the state
       OUT: mean (M) and variance (S) of the action
       """
        W = self.param("W", self.w_init, (self.action_dim, self.obs_dim))
        b = self.param("b", self.b_init, (1, self.action_dim))
        M = m @ W.T + b  # mean output
        S = W @ s @ W.T  # output variance
        V = W.T  # input output covariance

        # if squash:  # TODO do we ever not want to squash?
        M, S, V2 = self._squash_sin(M, S, self.max_action)
        V = V @ V2

        return M, S, V

    @staticmethod
    def _squash_sin(m, s, max_action=None):
        """
        Squashing function, passing the controls mean and variance
        through a sinus, as in gSin.m. The output is in [-max_action, max_action].
        IN: mean (m) and variance(s) of the control input, max_action
        OUT: mean (M) variance (S) and input-output (C) covariance of the squashed
             control input
        """
        k = jnp.shape(m)[1]
        max_action = max_action * jnp.ones((1, k))

        M = max_action * jnp.exp(-0.5 * jnp.diag(s)) * jnp.sin(m)

        lq = -0.5 * (jnp.diag(s)[:, None] + jnp.diag(s)[None, :])
        q = jnp.exp(lq)
        mT = jnp.transpose(m, (1, 0))
        S = (jnp.exp(lq + s) - q) * jnp.cos(mT - m) - (jnp.exp(lq - s) - q) * jnp.cos(mT + m)
        S = 0.5 * max_action * jnp.transpose(max_action, (1, 0)) * S

        diag_fn = lambda x: jnp.diag(x, k=0)
        C = max_action * jax.vmap(diag_fn)(jnp.exp(-0.5 * jnp.diag(s)) * jnp.cos(m))

        return M, S, C.reshape((k, k))

    # def randomise(self):
    #     mean = 0
    #     sigma = 1
    #     self.W.assign(mean + sigma * objax.random.normal(self.W.shape))
    #     self.b.assign(mean + sigma * objax.random.normal(self.b.shape))
    #
    # def randomize(self, key):
    #     """
    #     Randomize weights and biases using a JAX random key
    #     """
    #     w_key, b_key = jax.random.split(key)
    #
    #     # Randomize weights
    #     w_init = jax.random.normal(w_key, self.W.shape)
    #     b_init = jax.random.normal(b_key, self.b.shape)
    #
    #     self.W = w_init
    #     self.b = b_init