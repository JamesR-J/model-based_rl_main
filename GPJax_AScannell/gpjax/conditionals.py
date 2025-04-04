#!/usr/bin/env python3
from typing import Optional, Union

import jax
from jax import numpy as jnp
from jax import scipy as jsp

from GPJax_AScannell.gpjax.likelihoods import Likelihood
# from multidispatch import multifunction
from GPJax_AScannell.gpjax.multidispatch import multifunction
# from tensor_annotations import jax as tjax

from GPJax_AScannell.gpjax.config import default_jitter
from GPJax_AScannell.gpjax.custom_types import MeanAndCovariance
from jaxtyping import Array
from GPJax_AScannell.gpjax.kernels import Kernel, MultioutputKernel, SeparateIndependent


@multifunction(None, None, None, None, Kernel)
def conditional(kernel_params: dict,
                likelihood_params: dict,
                Xnew: Array,
                X: Array,
                # inducing_variable: InducingVariable,
                # TODO implement dispatching for inducing variables
                kernel: Kernel,
                f: Union[Array, Array],
                full_cov: Optional[bool] = False,
                full_output_cov: Optional[bool] = False,
                q_sqrt: Optional[Union[Array, Array, Array, Array]] = None,
                white: Optional[bool] = False) -> MeanAndCovariance:
    """GP Conditional.

    Multidispatch handles changing implementation for multioutput etc
    """
    f_mean, f_cov = single_output_conditional(kernel_params,
                                              likelihood_params,
                                              Xnew,
                                              # inducing_variable,
                                              X,
                                              kernel,
                                              f=f,
                                              full_cov=full_cov,
                                              q_sqrt=q_sqrt,
                                              white=white)
    return f_mean, f_cov


@multifunction(None, None, None, None, Kernel)
def _conditional(kernel_params: dict,
                xnew: Array,
                X: Array,
                # inducing_variable: InducingVariable,
                # TODO implement dispatching for inducing variables
                kernel: Kernel,
                f: Union[Array, Array],
                full_cov: Optional[bool] = False,
                full_output_cov: Optional[bool] = False,
                q_sqrt: Optional[Union[Array, Array, Array, Array]] = None,
                white: Optional[bool] = False) -> MeanAndCovariance:
                """GP Conditional for a single data point xnew [input_dim].

                Multidispatch handles changing implementation for multioutput etc
                """
                f_mean, f_cov = single_output_conditional(kernel_params,
                    xnew,
                    # inducing_variable,
                    X,
                    kernel,
                    f=f,
                    full_cov=full_cov,
                    q_sqrt=q_sqrt,
                    white=white,
                )
                return f_mean, f_cov


def single_output_conditional(kernel_params: dict,
                              likelihood_params: dict,
                              Xnew: Array,
                              X: Array,
                              # inducing_variable: InducingVariable,
                              kernel: Kernel,
                              f: Array,
                              full_cov: Optional[bool] = False,
                              full_output_cov: Optional[bool] = False,
                              q_sqrt: Optional[Union[Array, Array]] = None,
                              white: Optional[bool] = False) -> MeanAndCovariance:
    """Single-output GP conditional."""
    Kmm = (kernel(kernel_params, X, X) + jnp.eye(X.shape[-2], dtype=X.dtype) * default_jitter())  # [..., M, M]
    # Kmm = Kmm + jnp.eye(X.shape[-2], dtype=X.dtype) * 1.0  # [..., M, M]
    Kmm = Kmm + jnp.eye(X.shape[-2], dtype=X.dtype) * likelihood_params["variance"]  # [..., M, M]
    Kmn = kernel(kernel_params, X, Xnew)  # [M, N]
    Knn = kernel(kernel_params, Xnew, full_cov=full_cov)  # [N, N]

    # setup axis containing output dim which are to be mapped over
    if full_cov:  # [output_dim, num_data, num_data]
        out_axes = (-1, 0)
    else:  # [num_data, output_dim]
        out_axes = (-1, -1)
    if q_sqrt is not None:
        if q_sqrt.ndim == 2:
            in_axes = (-1, -1)
        elif q_sqrt.ndim == 3:
            in_axes = (-1, 0)
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))

        def base_conditional_wrapper(f_, q_sqrt_):
            return base_conditional(Kmn, Kmm, Knn, f_, full_cov=full_cov, q_sqrt=q_sqrt_, white=white)

        f_mean, f_cov = jax.vmap(base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes)(f, q_sqrt)
    else:
        def base_conditional_wrapper(f_):
            return base_conditional(Kmn, Kmm, Knn, f_, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

        f_mean, f_cov = jax.vmap(base_conditional_wrapper, in_axes=-1, out_axes=out_axes)(f)
    return f_mean, f_cov


# @conditional.dispatch(None, None, None, MultioutputKernel)
@conditional.dispatch(None, None, None, None, SeparateIndependent)
def independent_output_conditional(
    # def conditional(
    kernel_params: dict,
    likelihood_params: dict,
    Xnew: Array,
    X: Array,
    # inducing_variable: InducingVariable,
    kernel: Union[Kernel, MultioutputKernel],
    f: Array,
    full_cov: Optional[bool] = False,
    full_output_cov: Optional[bool] = False,
    q_sqrt: Optional[Union[Array, Array]] = None,
    white: Optional[bool] = False):
    """Multi-output GP conditional where outputs are assumed independent."""
    Kmm = (kernel(kernel_params, X, X) + jnp.eye(X.shape[-2], dtype=X.dtype) * default_jitter())  # [P, M, M]
    Kmm = Kmm + jnp.eye(X.shape[-2], dtype=X.dtype) * likelihood_params["variance"]  # [P, M, M]
    # TODO if i turn off the above it then works, but it doesn't optimise, the vice versa is true
    Kmn = kernel(kernel_params, X, Xnew)  # [P, M, N]
    Knn = kernel(kernel_params, Xnew, full_cov=full_cov)  # [P, N, N] or [N, P]

    # setup axis containing output dim which are to be mapped over
    if full_cov:
        # [output_dim, num_data, num_data]
        out_axes = (-1, 0)
    else:
        # [num_data, output_dim]
        out_axes = (-1, -1)
    if q_sqrt is not None:
        if q_sqrt.ndim == 2:
            if full_cov:
                in_axes = (0, 0, 0, -1, -1)
            else:
                in_axes = (0, 0, -1, -1, -1)
        elif q_sqrt.ndim == 3:
            if full_cov:
                in_axes = (0, 0, 0, -1, 0)
            else:
                in_axes = (0, 0, -1, -1, 0)
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))

        def base_conditional_wrapper(Kmn_, Kmm_, Knn_, f_, q_sqrt_):
            return base_conditional(
                Kmn_, Kmm_, Knn_, f_, full_cov=full_cov, q_sqrt=q_sqrt_, white=white
            )

        F_mean, F_cov = jax.vmap(base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes)(Kmn, Kmm, Knn, f, q_sqrt)
    else:
        def base_conditional_wrapper(Kmn_, Kmm_, Knn_, f_):
            return base_conditional(Kmn_, Kmm_, Knn_, f_, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

        if full_cov:
            in_axes = (0, 0, 0, -1)
        else:
            in_axes = (0, 0, -1, -1)
        F_mean, F_cov = jax.vmap(base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes)(Kmn, Kmm, Knn, f)
    return F_mean, F_cov


@_conditional.dispatch(None, None, None, SeparateIndependent)
def _independent_output_conditional(
    kernel_params: dict,
    xnew: Array,
    X: Array,
    # inducing_variable: InducingVariable,
    kernel: Union[Kernel, MultioutputKernel],
    f: Array,
    q_sqrt: Optional[
        Union[Array, Array]] = None,
    white: Optional[bool] = False,
):
    """Independent multi-output GP conditional for single input xnew [input_dim]"""
    Kmm = (
        kernel(kernel_params, X, X)
        + jnp.eye(X.shape[-2], dtype=X.dtype) * default_jitter()
    )  # [P, M, M]
    Kmn = kernel(kernel_params, X, xnew)  # [P, M]
    Knn = kernel(kernel_params, xnew, full_cov=False)  # [P]
    # print("inside _independent_output_conditional")
    # print(Kmm.shape)
    # print(Kmn.shape)
    # print(Knn.shape)

    # setup axis containing output dim which are to be mapped over
    out_axes = (-1, -1)
    if q_sqrt is not None:
        if q_sqrt.ndim == 2:
            in_axes = (0, 0, 0, -1, -1)
        elif q_sqrt.ndim == 3:
            in_axes = (0, 0, 0, -1, 0)
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))

        def base_conditional_wrapper(Kmn_, Kmm_, Knn_, f_, q_sqrt_):
            return base_conditional(
                Kmn_, Kmm_, Knn_, f_, full_cov=False, q_sqrt=q_sqrt_, white=white
            )

        F_mean, F_cov = jax.vmap(
            base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes
        )(Kmn, Kmm, Knn, f, q_sqrt)
    else:
        raise NotImplementedError("need to implement this")

        # def base_conditional_wrapper(Kmn_, Kmm_, Knn_, f_):
        #     return base_conditional(
        #         Kmn_, Kmm_, Knn_, f_, full_cov=full_cov, q_sqrt=q_sqrt, white=white
        #     )

        # if full_cov:
        #     in_axes = (0, 0, 0, -1)
        # else:
        #     in_axes = (0, 0, -1, -1)
        # F_mean, F_cov = jax.vmap(
        #     base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes
        # )(Kmn, Kmm, Knn, f)
    return F_mean, F_cov


def fully_correlated_conditional(
    kernel_params: dict,
    Xnew: Array,
    X: Array,
    # inducing_variable: InducingVariable,
    kernel: Union[Kernel, MultioutputKernel],
    f: Array,
    full_cov: Optional[bool] = False,
    full_output_cov: Optional[bool] = False,
    q_sqrt: Optional[Union[Array, Array]] = None,
    white: Optional[bool] = False,
):
    """Multi-output GP conditional where the conditioning points are fully correlated."""
    raise NotImplementedError("Still needs to be implemented")


def base_conditional(
    Kmn: Array,
    Kmm: Array,
    Knn: Union[Array, Array],
    f: Array,
    full_cov: Optional[bool] = False,
    q_sqrt: Optional[Union[Array, Array]] = None,
    white: Optional[bool] = False,
):
    r"""Base conditional for single outputs.

    Handling of output dimensions (independent/correlated) will be separate.

    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2; 0, Kmm)
      p(g1) = N(g1; 0, Knn)
      p(g1 | g2) = N(g1; Knm (Kmm⁻¹) g2, Knn - Knm (Kmm⁻¹) Kmn)
    And
      q(g2) = N(g2; f, q_sqrt q_sqrtᵀ)
    This method computes the mean and (co)variance of
      q(g1) = ∫ q(g2) p(g1 | g2)
    :param Kmn: [M, N]
    :param Kmm: [M, M]
    :param Knn: [N, N]  or  [N]
    :param f: [M]
    :param full_cov: bool
    :param q_sqrt: [M, M] (lower triangular) or [M] (diagonal)
    :param white: bool
    :return: mean [N] and (co)variance [N]  or [N, N]
    """
    Lm = jsp.linalg.cholesky(Kmm, lower=True)
    return base_conditional_with_lm(
        Kmn=Kmn, Lm=Lm, Knn=Knn, f=f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    )


def base_conditional_with_lm(
    Kmn: Array,
    Lm: Array,
    Knn: Union[Array, Array],  # n is new data, m is dataset
    f: Array,
    full_cov: Optional[bool] = False,
    q_sqrt: Optional[Union[Array, Array]] = None,
    white: Optional[bool] = False,
):
    """Same as base_conditional but expects the cholesky Lm instead of Kmm = Lm Lm.T

    Lm can be precomputed, improving performance.
    """
    A = jsp.linalg.solve_triangular(Lm, Kmn, lower=True)  # [M, N]

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - jnp.matmul(A.T, A)
    else:
        fvar = Knn - jnp.sum(jnp.square(A), 0)

    # another backsubstitution in the unwhitened case
    if not white:
        A = jsp.linalg.solve_triangular(Lm.T, A, lower=False)  # [M, N]

    # conditional mean
    fmean = A.T @ f  # [N]

    # covariance due to inducing variables
    if q_sqrt is not None:
        if q_sqrt.ndim == 1:
            LTA = jnp.expand_dims(q_sqrt, axis=-1) * A  # [M, N]
        elif q_sqrt.ndim == 2:
            LTA = q_sqrt.T @ A  # [M, N]
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))

        if full_cov:
            fvar = fvar + LTA.T @ LTA  # [N, N]
        else:
            fvar = fvar + jnp.sum(jnp.square(LTA), 0)  # [N]

    return fmean, fvar
