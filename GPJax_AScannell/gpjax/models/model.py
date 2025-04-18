#!/usr/bin/env python3
import abc
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
from GPJax_AScannell.gpjax.base import Module
from GPJax_AScannell.gpjax.custom_types import InputData, MeanAndCovariance
from GPJax_AScannell.gpjax.kernels import Kernel
from GPJax_AScannell.gpjax.likelihoods import Likelihood
from GPJax_AScannell.gpjax.mean_functions import MeanFunction, Zero
from GPJax_AScannell.gpjax.utilities.ops import sample_mvn_diag, sample_mvn
from GPJax_AScannell.gpjax.prediction import gp_predict_f
from jax import scipy as jsp
from GPJax_AScannell.gpjax.config import default_jitter
from functools import partial


jax.config.update("jax_enable_x64", True)



class GPModel(Module, abc.ABC):
    def __init__(self,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 mean_function: Optional[MeanFunction] = None,
                 num_latent_gps: int = None,
                 jitter=1e-6):
        assert num_latent_gps is not None, "GP requires specification of num_latent_gps"
        self.num_latent_gps = num_latent_gps
        self.kernel = kernel
        self.likelihood = likelihood
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function
        self.jitter = jitter

    @abc.abstractmethod
    def predict_f(self,
                  params: dict,
                  Xnew: InputData,
                  train_data: Optional[InputData] = None,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndCovariance:
        """Compute mean and (co)variance of latent function at Xnew.

        :param Xnew: inputs with shape [num_data, input_dim]
        :param full_cov:
            If True, draw correlated samples over Xnew. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            TODO Not implemented
        :returns: tuple of Tensors (mean, variance),
            means.shape == [num_data, output_dim],
            If full_cov=True and full_output_cov=False,
                var.shape == [output_dim, num_data, num_data]
            If full_cov=False,
                var.shape == [num_data, output_dim]
        """
        raise NotImplementedError

    def predict_f_samples(self,
                          params: dict,
                          key,
                          Xnew: InputData,
                          train_data: Optional[InputData] = None,
                          num_samples: int = 1,
                          full_cov: bool = False,
                          full_output_cov: bool = False) -> MeanAndCovariance:
                        """Draw samples from latent function posterior at Xnew.

                        :param params: dict of associated params
                        :param key : jax.random.PRNGKey()
                        :param Xnew: inputs with shape [num_data, input_dim]
                        :param num_samples: number of samples to draw
                        :param full_cov:
                            If True, draw correlated samples over Xnew. Computes the Cholesky over the
                            dense covariance matrix of size [num_data, num_data].
                            If False, draw samples that are uncorrelated over the inputs.
                        :param full_output_cov:
                            TODO Not implemented
                        :returns: samples with shape [num_samples, num_data, output_dim]
                        """
                        if full_output_cov:
                            raise NotImplementedError("full_output_cov=True is not implemented yet.")
                        f_mean, f_cov = self.predict_f(params, Xnew, train_data, full_cov, full_output_cov)

                        if f_cov.ndim == 3:
                            samples = sample_mvn(key, f_mean.T, f_cov, num_samples)
                        elif f_cov.ndim == 2:
                            samples = sample_mvn_diag(key, f_mean.T, jnp.sqrt(f_cov.T), num_samples)
                        else:
                            raise NotImplementedError("Bad dimension for f_cov")
                        return jnp.transpose(samples, [1, 2, 0])

    def predict_y(self,
                  params: dict,
                  Xnew: InputData,
                  train_data: Optional[InputData] = None,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndCovariance:
        """Compute the mean and (co)variance of function at Xnew."""
        if full_cov or full_output_cov:
            raise NotImplementedError(
                "The predict_y method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        f_mean, f_cov = self.predict_f(params, Xnew, train_data, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_mean_and_var(params["likelihood"], f_mean, f_cov)


class GPR(GPModel):
    def __init__(self,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 mean_function: Optional[MeanFunction] = None,
                 num_latent_gps: Optional[int] = 1):
        super().__init__(kernel=kernel,
                         likelihood=likelihood,
                         mean_function=mean_function,
                         num_latent_gps=num_latent_gps)

    def get_params(self, q_mu=None) -> dict:
        kernel_params = self.kernel.get_params()
        likelihood_params = self.likelihood.get_params()
        mean_function_params = self.mean_function.get_params()
        return {'kernel': kernel_params,
                'likelihood': likelihood_params,
                "mean_function": mean_function_params}

    def get_transforms(self) -> dict:
        kernel_transforms = self.kernel.get_transforms()
        if self.likelihood is not None:
            likelihood_transforms = self.likelihood.get_transforms()
        else:
            likelihood_transforms = {}
        mean_function_transforms = self.mean_function.get_transforms()

        return {
            "kernel": kernel_transforms,
            "likelihood": likelihood_transforms,
            "mean_function": mean_function_transforms,
        }

    def predict_f(self,
                  params: dict,
                  Xnew: InputData,
                  train_data: Optional[InputData] = None,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndCovariance:
        """Compute mean and (co)variance of latent function at Xnew.

        :param Xnew: inputs with shape [num_data, input_dim]
        :param full_cov:
            If True, draw correlated samples over Xnew. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            TODO Not implemented
        :returns: tuple of Tensors (mean, variance),
            means.shape == [num_data, output_dim],
            If full_cov=True and full_output_cov=False,
                var.shape == [output_dim, num_data, num_data]
            If full_cov=False,
                var.shape == [num_data, output_dim]
        """

        err = train_data.y - self.mean_function(params["mean_function"], train_data.X)

        return gp_predict_f(params, Xnew, train_data.X, self.kernel, self.mean_function, err,
                            full_cov, full_output_cov, None, False)

    @partial(jax.jit, static_argnums=(0,))
    def log_marginal_likelihood(self, params, data):
        x = data.X
        y = data.y
        Kmm = self.kernel(params["kernel"], x, x) + jnp.eye(x.shape[-2], dtype=x.dtype) * default_jitter()
        Kmm += jnp.eye(x.shape[-2], dtype=x.dtype) * params["likelihood"]["variance"]
        Lm = jsp.linalg.cholesky(Kmm, lower=True)
        mx = self.mean_function(params["mean_function"], x)

        def tf_multivariate_normal(x, mu, L):
            """
            Computes the log-density of a multivariate normal.
            :param x: sample(s) for which we want the density
            :param mu: mean(s) of the normal distribution
            :param L: Cholesky decomposition of the covariance matrix
            :return: log densities
            """
            d = x - mu
            alpha = jsp.linalg.solve_triangular(L, d, lower=True)
            num_dims = d.shape[0]
            p = -0.5 * jnp.sum(jnp.square(alpha), axis=0)
            p -= 0.5 * num_dims * jnp.log(2 * jnp.pi)
            p -= jnp.sum(jnp.log(jnp.diag(L)))
            return p

        log_prob = tf_multivariate_normal(y, mx, Lm)

        return jnp.sum(log_prob)

    @partial(jax.jit, static_argnums=(0,))
    def multi_output_log_marginal_likelihood(self, params, data):
        x = data.X
        y = data.y
        Kmm = self.kernel.kernels[0](params["kernel"], x, x) + jnp.eye(x.shape[-2], dtype=x.dtype) * default_jitter()
        # TODO a dodgy fix that assumes the kernels are the same for each dimension
        Kmm += jnp.eye(x.shape[-2], dtype=x.dtype) * params["likelihood"]["variance"]
        Lm = jsp.linalg.cholesky(Kmm, lower=True)
        mx = self.mean_function(params["mean_function"], x)
        # TODO added in below which is a dodgy fix that only works if mean function is zero
        mx = jnp.expand_dims(mx[:, 0], axis=-1)

        def tf_multivariate_normal(x, mu, L):
            """
            Computes the log-density of a multivariate normal.
            :param x: sample(s) for which we want the density
            :param mu: mean(s) of the normal distribution
            :param L: Cholesky decomposition of the covariance matrix
            :return: log densities
            """
            d = x - mu
            alpha = jsp.linalg.solve_triangular(L, d, lower=True)
            num_dims = d.shape[0]
            p = -0.5 * jnp.sum(jnp.square(alpha), axis=0)
            p -= 0.5 * num_dims * jnp.log(2 * jnp.pi)
            p -= jnp.sum(jnp.log(jnp.diag(L)))
            return p

        log_prob = tf_multivariate_normal(y, mx, Lm)

        return jnp.sum(log_prob)