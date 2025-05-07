# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Optional, Union

from GPJax_AScannell.gpjax.config import Config, default_float
from GPJax_AScannell.gpjax.custom_types import Covariance, Input1, Input2
from GPJax_AScannell.gpjax.kernels import Kernel, covariance_decorator
from GPJax_AScannell.gpjax.kernels.distances import scaled_squared_euclidean_distance
from jax import numpy as jnp
from jaxtyping import Array


def linear_kern_fn(params: dict, x1: Array, x2: Array = None) -> jnp.float64:
    variance = params["variance"]
    if x2 is None:
        return jnp.matmul(x1 * variance, x1.T)
    else:
        return jnp.tensordot(x1 * variance, x2, [[-1], [-1]])


@covariance_decorator
def linear_cov_fn(params:dict, X1: Input1, X2: Input2 = None) -> Covariance:
    return linear_kern_fn(params, X1, X2)



class Linear(Kernel, abc.ABC):
    """
    The linear kernel. Functions drawn from a GP with this kernel are linear, i.e. f(x) = cx.
    The kernel equation is

        k(x, y) = σ²xy

    where σ² is the variance parameter.
    """
    def __init__(
        self, variance: Optional[jnp.float64] = 1.0,
              name: Optional[str] = "Linear kernel",):
        """
        :param variance: the (initial) value for the variance parameter(s),
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param active_dims: a slice or list specifying which columns of X are used
        """
        super().__init__(name=name)
        self.variance = jnp.array(variance)

    def get_params(self) -> dict:
        return {"variance": self.variance}

    def get_transforms(self) -> dict:
        return {
            "variance": Config.positive_bijector,
        }

    @staticmethod
    def K(params:dict, X1: Input1, X2: Input2 = None) -> Covariance:
        return linear_cov_fn(params, X1, X2)

    # @inherit_check_shapes
    # def K_diag(self, X: TensorType) -> tf.Tensor:
    #     return tf.reduce_sum(tf.square(X) * self.variance, axis=-1)


# class Polynomial(Linear):
#     """
#     The Polynomial kernel. Functions drawn from a GP with this kernel are
#     polynomials of degree `d`. The kernel equation is
#
#         k(x, y) = (σ²xy + γ)ᵈ
#
#     where:
#     σ² is the variance parameter,
#     γ is the offset parameter,
#     d is the degree parameter.
#     """
#
#     @check_shapes(
#         "variance: [broadcast n_active_dims]",
#     )
#     def __init__(
#         self,
#         degree: TensorType = 3.0,
#         variance: TensorType = 1.0,
#         offset: TensorType = 1.0,
#         active_dims: Optional[ActiveDims] = None,
#     ) -> None:
#         """
#         :param degree: the degree of the polynomial
#         :param variance: the (initial) value for the variance parameter(s),
#             to induce ARD behaviour this must be initialised as an array the same
#             length as the the number of active dimensions e.g. [1., 1., 1.]
#         :param offset: the offset of the polynomial
#         :param active_dims: a slice or list specifying which columns of X are used
#         """
#         super().__init__(variance, active_dims)
#         self.degree = degree
#         self.offset = Parameter(offset, transform=positive())
#
#     @inherit_check_shapes
#     def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
#         return (super().K(X, X2) + self.offset) ** self.degree
#
#     @inherit_check_shapes
#     def K_diag(self, X: TensorType) -> tf.Tensor:
#         return (super().K_diag(X) + self.offset) ** self.degree
