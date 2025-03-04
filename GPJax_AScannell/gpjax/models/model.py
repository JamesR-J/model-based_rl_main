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
import beartype.typing as tp

from cola.annotations import PSD
from cola.linalg.algorithm_base import Algorithm
from cola.linalg.decompositions.decompositions import Cholesky
from cola.linalg.inverse.inv import solve
from cola.ops.operators import I_like

from gpjax.kernels.base import AbstractKernel
from gpjax.likelihoods import AbstractLikelihood, Gaussian, NonGaussian
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.kernels import RFF
from gpjax.typing import Array, FunctionalSample, KeyArray
# from jaxtyping import FLoat
from flax import nnx


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
        if train_data is None:
            gp_data = params["train_data"]
        else:
            gp_data = train_data
        return gp_predict_f(params, Xnew, gp_data, self.kernel, self.mean_function, params["q_mu"],
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

    # def _build_fourier_features_fn(self,
    #                                kernel,
    #                                num_features: int,
    #                                key: KeyArray):
    #     r"""Return a function that evaluates features sampled from the Fourier feature
    #     decomposition of the prior's kernel.
    #
    #     Args:
    #         prior (Prior): The Prior distribution.
    #         num_features (int): The number of feature functions to be sampled.
    #         key (KeyArray): The random seed used.
    #
    #     Returns
    #     -------
    #         Callable: A callable function evaluation the sampled feature functions.
    #     """
    #     if (not isinstance(num_features, int)) or num_features <= 0:
    #         raise ValueError("num_features must be a positive integer")
    #
    #     # Approximate kernel with feature decomposition
    #     approximate_kernel = RFF(base_kernel=kernel, num_basis_fns=num_features, key=key)
    #
    #     def eval_fourier_features(test_inputs):
    #         Phi = approximate_kernel.compute_features(x=test_inputs)
    #         Phi *= jnp.sqrt(kernel.variance.value / num_features)
    #         return Phi
    #
    #     return eval_fourier_features
    #
    # def predict_f_sample_approx(self,
    #                             params: dict,
    #                             key,
    #                             Xnew: InputData,
    #                             num_samples: int = 1,
    #                             num_features: int | None = 100,
    #                             solver_algorithm: tp.Optional[Algorithm] = Cholesky(),
    #                             full_cov: bool = False,
    #                             full_output_cov: bool = False) -> FunctionalSample:
    #     r"""Draw approximate samples from the Gaussian process posterior.
    #
    #     Build an approximate sample from the Gaussian process posterior. This method
    #     provides a function that returns the evaluations of a sample across any given
    #     inputs.
    #
    #     Unlike when building approximate samples from a Gaussian process prior, decompositions
    #     based on Fourier features alone rarely give accurate samples. Therefore, we must also
    #     include an additional set of features (known as canonical features) to better model the
    #     transition from Gaussian process prior to Gaussian process posterior. For more details
    #     see [Wilson et. al. (2020)](https://arxiv.org/abs/2002.09309).
    #
    #     In particular, we approximate the Gaussian processes' posterior as the finite
    #     feature approximation
    #     $\hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i + \sum{j=1}^N v_jk(.,x_j)$
    #     where $\phi_i$ are m features sampled from the Fourier feature decomposition of
    #     the model's kernel and $k(., x_j)$ are N canonical features. The Fourier
    #     weights $\theta_i$ are samples from a unit Gaussian. See
    #     [Wilson et. al. (2020)](https://arxiv.org/abs/2002.09309) for expressions
    #     for the canonical weights $v_j$.
    #
    #     A key property of such functional samples is that the same sample draw is
    #     evaluated for all queries. Consistency is a property that is prohibitively costly
    #     to ensure when sampling exactly from the GP prior, as the cost of exact sampling
    #     scales cubically with the size of the sample. In contrast, finite feature representations
    #     can be evaluated with constant cost regardless of the required number of queries.
    #
    #     Args:
    #         num_samples (int): The desired number of samples.
    #         key (KeyArray): The random seed used for the sample(s).
    #         num_features (int): The number of features used when approximating the
    #             kernel.
    #         solver_algorithm (Optional[Algorithm], optional): The algorithm to use for the solves of
    #             the inverse of the covariance matrix. See the
    #             [CoLA documentation](https://cola.readthedocs.io/en/latest/package/cola.linalg.html#algorithms)
    #             for which solver to pick. For PSD matrices, CoLA currently recommends Cholesky() for small
    #             matrices and CG() for larger matrices. Select Auto() to let CoLA decide. Defaults to Cholesky().
    #
    #     Returns:
    #         FunctionalSample: A function representing an approximate sample from the Gaussian
    #         process prior.
    #     """
    #     if (not isinstance(num_samples, int)) or num_samples <= 0:
    #         raise ValueError("num_samples must be a positive integer")
    #
    #     # sample fourier features
    #     fourier_feature_fn = self._build_fourier_features_fn(self.kernel, num_features, key)
    #
    #     # sample fourier weights
    #     fourier_weights = jrandom.normal(key, [num_samples, 2 * num_features])  # [B, L]
    #
    #     # sample weights v for canonical features
    #     # v = Σ⁻¹ (y + ε - ɸ⍵) for  Σ = Kxx + Io² and ε ᯈ N(0, o²)
    #     obs_var = self.likelihood.obs_stddev.value ** 2
    #     Kxx = self.kernel.gram(params["train_data"])  # [N, N]
    #     Sigma = Kxx + I_like(Kxx) * (obs_var + self.jitter)  # [N, N]
    #     eps = jnp.sqrt(obs_var) * jrandom.normal(key, [train_data.n, num_samples])  # [N, B]
    #     y = train_data.y - self.mean_function(train_data.X)  # account for mean
    #     Phi = fourier_feature_fn(train_data.X)
    #     canonical_weights = solve(Sigma, y + eps - jnp.inner(Phi, fourier_weights), solver_algorithm)  # [N, B]
    #
    #     def sample_fn(test_inputs: Float[Array, "n D"]) -> Float[Array, "n B"]:
    #         fourier_features = fourier_feature_fn(test_inputs)  # [n, L]
    #         weight_space_contribution = jnp.inner(fourier_features, fourier_weights)  # [n, B]
    #         canonical_features = self.kernel.cross_covariance(test_inputs, train_data.X)  # [n, N]
    #         function_space_contribution = jnp.matmul(canonical_features, canonical_weights)
    #
    #         return (self.mean_function(test_inputs) + weight_space_contribution + function_space_contribution)
    #
    #     return sample_fn


class GPR(GPModel):
    def get_params(self) -> dict:
        kernel_params = self.kernel.get_params()
        likelihood_params = self.likelihood.get_params()
        mean_function_params = self.mean_function.get_params()
        return {'kernel':kernel_params,
                'likelihood': likelihood_params,
                "mean_function": mean_function_params}
