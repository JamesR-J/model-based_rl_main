#!/usr/bin/env python3
import jax.numpy as jnp
from GPJax_AScannell.gpjax import logdensities
from GPJax_AScannell.gpjax.config import default_float
from GPJax_AScannell.gpjax.likelihoods.base import ScalarLikelihood
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors


class Gaussian(ScalarLikelihood):
    """Gaussian likelihood with constant variance

    Small variances can lead to numerical instability during optimization so
    a lower bound of 1e-6 is imposed on the variance by default.
    """

    def __init__(self, variance=1.0, variance_lower_bound=1e-6):
        """
        :param variance: the noise variance; greater than variance_lower_bound
        :param variance_lower_bound: lower bound on the noise variance
        """
        super().__init__()

        if jnp.any(variance <= variance_lower_bound):
            raise ValueError(
                f"The variance of the Gaussian likelihood must be strictly greater than {variance_lower_bound}"
            )
        if not isinstance(variance, jnp.ndarray):
            self.variance = jnp.array([variance], dtype=default_float())
        else:
            self.variance = variance
        self.positive_bijector = tfb.Chain(
            [
                tfb.Shift(
                    jnp.ones(self.variance.shape, dtype=default_float())
                    * variance_lower_bound
                ),
                tfb.Softplus(),
            ]
        )

    def get_params(self) -> dict:
        return {"variance": self.variance}

    def get_transforms(self) -> dict:
        return {"variance": self.positive_bijector}

    def _scalar_log_prob(self, params: dict, F, Y):
        return logdensities.gaussian(Y, F, params["variance"])

    def conditional_mean(self, params: dict, F):
        # TODO this should make copy?
        return F
        # return jnp.identity(F)

    def conditional_variance(self, params: dict, F):
        return params["variance"] * jnp.ones(F.shape)
        # return params["variance"]
        # return jnp.fill(jnp.shape(F), jnp.squeeze(params["variance"]))
        # return jnp.fill(jnp.shape(F), jnp.squeeze(params["variance"]))

    def predict_mean_and_var(self, params: dict, Fmu, Fvar):
        return Fmu, Fvar + params["variance"]

    def predict_log_density(self, params: dict, Fmu, Fvar, Y):
        return jnp.sum(logdensities.gaussian(Y, Fmu, Fvar + params["variance"]), axis=-1)

    def variational_expectations(self, params, Fmu, Fvar, Y):
        variance = params["variance"]
        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * jnp.log(variance)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / variance,
            axis=-1,
        )
