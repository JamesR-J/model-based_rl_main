#!/usr/bin/env python3
from typing import Optional, Tuple, Union

# import tensor_annotations.jax as tjax
from jax import numpy as jnp
# from tensor_annotations.axes import Batch
from jaxtyping import Array

# from .shapes import N1, N2, InputDim, NumData, OutputDim, NumInducing

# MeanAndVariance = Tuple[jnp.ndarray, jnp.ndarray]
InputData = jnp.ndarray
OutputData = jnp.ndarray
# InputData = Array
# OutputData = Array
# MeanFunc = jnp.float64

# Variance = Union[jnp.float64, jnp.ndarray]
# Lengthscales = Union[jnp.float64, jnp.ndarray]


# Data types
SingleInput = Array
# Inputs = Array[NumData, InputDim]
# BatchedInputs = Array[Batch, NumData, InputDim]
# Output = Array[OutputDim]
# Outputs = Array[NumData, OutputDim]
# BatchedOutputs = Array[Batch, NumData, OutputDim]

# AnyInput = Union[Input, Inputs, BatchedInputs]
# AnyOutput = Union[Output, Outputs, BatchedOutputs]


# Kernel parameter types
Lengthscales = Union[Array, jnp.float64]
KernVariance = Array

Input1 = Union[
    Array,
    Array,
    Array,
]
Input2 = Optional[
    Union[
        Array,
        Array,
        Array,
    ]
]

Covariance = Union[
    Array,
    Array,
    Array,  # if X2=None
    Array,
    Array,  # if X2=None
]

MultiOutputCovariance = Union[
    Array,
    Array,
    Array,  # if X2=None
    Array,
    Array,  # if X2=None
    # tjax.Array5[Batch, OutputDim, N1, OutputDim, N2],
    # tjax.Array5[Batch, OutputDim, N1, OutputDim, N1],  # if X2=None
]

# Data types
Mean = Array
Variance = Array
# Covariance = Array[OutputDim, NumData, NumData]
# MeanAndVariance = Union[Tuple[Mean, Variance], Tuple[Mean, Covariance]]
MeanAndCovariance = Union[Tuple[Mean, Variance], Tuple[Mean, Covariance]]

InducingVariable = Array