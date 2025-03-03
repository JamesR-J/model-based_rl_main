#!/usr/bin/env python3
import abc
from typing import Optional

import jax
import jax.numpy as jnp
from GPJax_AScannell.gpjax.base import Module
from GPJax_AScannell.gpjax.custom_types import InputData, MeanAndCovariance
from GPJax_AScannell.gpjax.kernels import Kernel
from GPJax_AScannell.gpjax.likelihoods import Likelihood
from GPJax_AScannell.gpjax.mean_functions import MeanFunction, Zero
from GPJax_AScannell.gpjax.utilities.ops import sample_mvn_diag, sample_mvn
from GPJax_AScannell.gpjax.prediction import gp_predict_f

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

    def get_params(self, q_mu=None) -> dict:
        kernel_params = self.kernel.get_params()
        likelihood_params = self.likelihood.get_params()
        mean_function_params = self.mean_function.get_params()
        if q_mu is None:
            q_mu = jnp.zeros((1, self.num_latent_gps))  # dims = data points, num_gp
        q_mu = jnp.array(q_mu)
        return {'kernel':kernel_params,
                'likelihood': likelihood_params,
                "mean_function": mean_function_params,
                "q_mu": q_mu,}  # TODO last one is dodgy replacement, does it actually work?

    # @abc.abstractmethod
    def predict_f(self,
                  params: dict,
                  Xnew: InputData,
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
        return gp_predict_f(params, Xnew, params["train_data"], self.kernel, self.mean_function, params["q_mu"],
                            full_cov, full_output_cov, None, False)

    def predict_f_samples(self,
                          params: dict,
                          key,
                          Xnew: InputData,
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
                        f_mean, f_cov = self.predict_f(params, Xnew, full_cov, full_output_cov)

                        if f_cov.ndim == 3:
                            print("here")
                            print(f_mean.shape)
                            print(f_cov.shape)
                            samples = sample_mvn(key, f_mean.T, f_cov, num_samples)
                        elif f_cov.ndim == 2:
                            samples = sample_mvn_diag(key, f_mean.T, jnp.sqrt(f_cov.T), num_samples)
                        else:
                            raise NotImplementedError("Bad dimension for f_cov")
                        return jnp.transpose(samples, [1, 2, 0])

    def predict_f_samples_func(self,
                               params: dict,
                               key,
                               Xnew: InputData,
                               num_samples: int = 1,
                               full_cov: bool = False,
                               full_output_cov: bool = False):
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
                        f_mean, f_cov = self.predict_f(params, Xnew, full_cov, full_output_cov)

                        if f_cov.ndim == 3:
                            print("here")
                            print(f_mean.shape)
                            print(f_cov.shape)
                            samples = sample_mvn(key, f_mean.T, f_cov, num_samples)
                        elif f_cov.ndim == 2:
                            samples = sample_mvn_diag(key, f_mean.T, jnp.sqrt(f_cov.T), num_samples)
                        else:
                            raise NotImplementedError("Bad dimension for f_cov")
                        return jnp.transpose(samples, [1, 2, 0])

    def predict_y(self,
                  params: dict,
                  Xnew: InputData,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndCovariance:
        """Compute the mean and (co)variance of function at Xnew."""
        if full_cov or full_output_cov:
            raise NotImplementedError(
                "The predict_y method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        f_mean, f_cov = self.predict_f(params, Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_mean_and_var(params["likelihood"], f_mean, f_cov)


class GPR(GPModel):
    def get_params(self) -> dict:
        kernel_params = self.kernel.get_params()
        likelihood_params = self.likelihood.get_params()
        mean_function_params = self.mean_function.get_params()
        return {'kernel':kernel_params,
                'likelihood': likelihood_params,
                "mean_function": mean_function_params}
