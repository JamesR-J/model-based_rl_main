#!/usr/bin/env python3
from typing import Optional

from GPJax_AScannell.gpjax.config import default_float
from jax import numpy as jnp
import jax


jax.config.update("jax_enable_x64", True)


def scaled_squared_euclidean_distance(x1, x2, lengthscales: Optional = jnp.array([1.0], dtype=default_float())):
    """Returns ‖(X1 - X2ᵀ) / ℓ‖², i.e. the squared L₂-norm.

    :param x1: Single input array [input_dim]
    :param x2: Optional single input array [input_dim]
    :param lengthscales: lengthscale parameter, either [input_dim] or [1]
    :returns: covariance matrix [1]
    """
    scaled_diff = (x1 - x2) / lengthscales
    squared_euclidean_distance = jnp.dot(scaled_diff, scaled_diff.T)
    return jnp.squeeze(squared_euclidean_distance)
