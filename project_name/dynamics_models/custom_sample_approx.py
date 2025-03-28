from abc import abstractmethod
import jax
import beartype.typing as tp
from cola.annotations import PSD
from cola.linalg.algorithm_base import Algorithm
from cola.linalg.decompositions.decompositions import Cholesky
from cola.linalg.inverse.inv import solve
from cola.ops.operators import I_like
from flax import nnx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Float,
    Num,
)

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.kernels import RFF
from gpjax.kernels.base import AbstractKernel
from gpjax.likelihoods import (
    AbstractLikelihood,
    Gaussian,
    NonGaussian,
)
from gpjax.lower_cholesky import lower_cholesky
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.parameters import (
    Parameter,
    Real,
    Static,
)
from gpjax.typing import (
    Array,
    FunctionalSample,
    KeyArray,
)
from gpjax.gps import Prior
import beartype.typing as tp
import jax.random as jr
from jaxtyping import Float

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import BasisFunctionComputation
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.parameters import Static
from gpjax.typing import (
    Array,
    KeyArray,
)
import jax.random as jrandom
from functools import partial

K = tp.TypeVar("K", bound=AbstractKernel)
M = tp.TypeVar("M", bound=AbstractMeanFunction)
L = tp.TypeVar("L", bound=AbstractLikelihood)
NGL = tp.TypeVar("NGL", bound=NonGaussian)
GL = tp.TypeVar("GL", bound=Gaussian)


def adj_sample_approx_old(posterior,
        num_samples: int,
        train_data: Dataset,
        key: KeyArray,
        num_features: int | None = 100,
        solver_algorithm: tp.Optional[Algorithm] = Cholesky(),
):
    r"""Draw approximate samples from the Gaussian process posterior.

    Build an approximate sample from the Gaussian process posterior. This method
    provides a function that returns the evaluations of a sample across any given
    inputs.

    Unlike when building approximate samples from a Gaussian process prior, decompositions
    based on Fourier features alone rarely give accurate samples. Therefore, we must also
    include an additional set of features (known as canonical features) to better model the
    transition from Gaussian process prior to Gaussian process posterior. For more details
    see [Wilson et. al. (2020)](https://arxiv.org/abs/2002.09309).

    In particular, we approximate the Gaussian processes' posterior as the finite
    feature approximation
    $\hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i + \sum{j=1}^N v_jk(.,x_j)$
    where $\phi_i$ are m features sampled from the Fourier feature decomposition of
    the model's kernel and $k(., x_j)$ are N canonical features. The Fourier
    weights $\theta_i$ are samples from a unit Gaussian. See
    [Wilson et. al. (2020)](https://arxiv.org/abs/2002.09309) for expressions
    for the canonical weights $v_j$.

    A key property of such functional samples is that the same sample draw is
    evaluated for all queries. Consistency is a property that is prohibitively costly
    to ensure when sampling exactly from the GP prior, as the cost of exact sampling
    scales cubically with the size of the sample. In contrast, finite feature representations
    can be evaluated with constant cost regardless of the required number of queries.

    Args:
        num_samples (int): The desired number of samples.
        key (KeyArray): The random seed used for the sample(s).
        num_features (int): The number of features used when approximating the
            kernel.
        solver_algorithm (Optional[Algorithm], optional): The algorithm to use for the solves of
            the inverse of the covariance matrix. See the
            [CoLA documentation](https://cola.readthedocs.io/en/latest/package/cola.linalg.html#algorithms)
            for which solver to pick. For PSD matrices, CoLA currently recommends Cholesky() for small
            matrices and CG() for larger matrices. Select Auto() to let CoLA decide. Defaults to Cholesky().

    Returns:
        FunctionalSample: A function representing an approximate sample from the Gaussian
        process prior.
    """
    if (not isinstance(num_samples, int)) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    # sample fourier features
    fourier_feature_fn = _build_fourier_features_fn(posterior.prior, num_features, key)

    # sample fourier weights
    # fourier_weights = _build_fourier_weights(train_data.y.shape[0], num_samples, num_features, key)
    fourier_weights = jr.normal(key, [num_samples, 2 * num_features])  # [B, L]

    # sample weights v for canonical features
    # v = Σ⁻¹ (y + ε - ɸ⍵) for  Σ = Kxx + Io² and ε ᯈ N(0, o²)
    obs_var = posterior.likelihood.obs_stddev.value ** 2
    Kxx = posterior.prior.kernel.gram(train_data.X)  # [N, N]
    Sigma = Kxx + I_like(Kxx) * (obs_var + posterior.jitter)  # [N, N]
    eps = jnp.sqrt(obs_var) * jr.normal(key, [train_data.n, num_samples]) # [N, B]
    y = train_data.y - posterior.prior.mean_function(train_data.X)  # account for mean
    Phi = fourier_feature_fn(train_data.X)
    canonical_weights = solve(Sigma, y + eps - jnp.inner(Phi, fourier_weights), solver_algorithm)  # [N, B]

    def sample_fn(test_inputs: Float[Array, "n D"]) -> Float[Array, "n B"]:
        fourier_features = fourier_feature_fn(test_inputs)  # [n, L]
        weight_space_contribution = jnp.inner(fourier_features, fourier_weights)  # [n, B]
        canonical_features = posterior.prior.kernel.cross_covariance(test_inputs, train_data.X)  # [n, N]
        function_space_contribution = jnp.matmul(canonical_features, canonical_weights)

        return (posterior.prior.mean_function(test_inputs)
                + weight_space_contribution
                + function_space_contribution)

    return sample_fn

def adj_sample_approx(posterior,
        num_samples: int,
        train_data: Dataset,
        key: KeyArray,
        num_features: int | None = 100,
        solver_algorithm: tp.Optional[Algorithm] = Cholesky(),
) -> FunctionalSample:
    r"""Draw approximate samples from the Gaussian process posterior.

    Build an approximate sample from the Gaussian process posterior. This method
    provides a function that returns the evaluations of a sample across any given
    inputs.

    Unlike when building approximate samples from a Gaussian process prior, decompositions
    based on Fourier features alone rarely give accurate samples. Therefore, we must also
    include an additional set of features (known as canonical features) to better model the
    transition from Gaussian process prior to Gaussian process posterior. For more details
    see [Wilson et. al. (2020)](https://arxiv.org/abs/2002.09309).

    In particular, we approximate the Gaussian processes' posterior as the finite
    feature approximation
    $\hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i + \sum{j=1}^N v_jk(.,x_j)$
    where $\phi_i$ are m features sampled from the Fourier feature decomposition of
    the model's kernel and $k(., x_j)$ are N canonical features. The Fourier
    weights $\theta_i$ are samples from a unit Gaussian. See
    [Wilson et. al. (2020)](https://arxiv.org/abs/2002.09309) for expressions
    for the canonical weights $v_j$.

    A key property of such functional samples is that the same sample draw is
    evaluated for all queries. Consistency is a property that is prohibitively costly
    to ensure when sampling exactly from the GP prior, as the cost of exact sampling
    scales cubically with the size of the sample. In contrast, finite feature representations
    can be evaluated with constant cost regardless of the required number of queries.

    Args:
        num_samples (int): The desired number of samples.
        key (KeyArray): The random seed used for the sample(s).
        num_features (int): The number of features used when approximating the
            kernel.
        solver_algorithm (Optional[Algorithm], optional): The algorithm to use for the solves of
            the inverse of the covariance matrix. See the
            [CoLA documentation](https://cola.readthedocs.io/en/latest/package/cola.linalg.html#algorithms)
            for which solver to pick. For PSD matrices, CoLA currently recommends Cholesky() for small
            matrices and CG() for larger matrices. Select Auto() to let CoLA decide. Defaults to Cholesky().

    Returns:
        FunctionalSample: A function representing an approximate sample from the Gaussian
        process prior.
    """
    if (not isinstance(num_samples, int)) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    # sample fourier features
    fourier_feature_fn = _build_fourier_features_fn(posterior.prior, num_features, key)

    # sample fourier weights
    # fourier_weights = _build_fourier_weights(train_data.y.shape[0], num_samples, num_features, key)
    fourier_weights = jr.normal(key, [num_samples, 2 * num_features])  # [B, L]

    # sample weights v for canonical features
    # v = Σ⁻¹ (y + ε - ɸ⍵) for  Σ = Kxx + Io² and ε ᯈ N(0, o²)
    obs_var = posterior.likelihood.obs_stddev.value ** 2
    Kxx = posterior.prior.kernel.gram(train_data.X)  # [N, N]
    Sigma = Kxx + I_like(Kxx) * (obs_var + posterior.jitter)  # [N, N]
    eps = jnp.sqrt(obs_var) * jr.normal(key, [train_data.n, num_samples]) # [N, B]
    y = train_data.y - posterior.prior.mean_function(train_data.X)  # account for mean
    Phi = fourier_feature_fn(train_data.X)
    canonical_weights = solve(Sigma, y + eps - jnp.inner(Phi, fourier_weights), solver_algorithm)  # [N, B]

    @partial(jax.jit, static_argnums=(1, 2))
    def sample_fn(test_inputs, un1, un2, un3, un4, un5) -> Float[Array, "n B"]:
        num_points = test_inputs.shape[0]
        out_dim = test_inputs.shape[1]
        label = jnp.tile(jnp.array(jnp.linspace(0, out_dim - 1, out_dim)), num_points)
        test_inputs = jnp.hstack((jnp.repeat(test_inputs, repeats=out_dim, axis=0), jnp.expand_dims(label, axis=-1)))
        fourier_features = fourier_feature_fn(test_inputs)  # [n, L]
        weight_space_contribution = jnp.inner(fourier_features, fourier_weights)  # [n, B]
        canonical_features = posterior.prior.kernel.cross_covariance(test_inputs, train_data.X)  # [n, N]
        function_space_contribution = jnp.matmul(canonical_features, canonical_weights)

        output = (posterior.prior.mean_function(test_inputs)
                + weight_space_contribution
                + function_space_contribution)

        return jnp.squeeze(output, axis=-1)

    return sample_fn


def _build_fourier_features_fn(
    prior: Prior, num_features: int, key: KeyArray
) -> tp.Callable[[Float[Array, "N D"]], Float[Array, "N L"]]:
    r"""Return a function that evaluates features sampled from the Fourier feature
    decomposition of the prior's kernel.

    Args:
        prior (Prior): The Prior distribution.
        num_features (int): The number of feature functions to be sampled.
        key (KeyArray): The random seed used.

    Returns
    -------
        Callable: A callable function evaluation the sampled feature functions.
    """
    if (not isinstance(num_features, int)) or num_features <= 0:
        raise ValueError("num_features must be a positive integer")

    # Approximate kernel with feature decomposition
    approximate_kernel = MO_RFF(kernel0=prior.kernel.kernel0, kernel1=prior.kernel.kernel1, num_basis_fns=num_features, key=key)

    def eval_fourier_features(test_inputs: Float[Array, "N D"]) -> Float[Array, "N L"]:
        Phi = approximate_kernel.compute_features(x=test_inputs)
        return Phi

    return eval_fourier_features

def _build_fourier_weights(dataset_shape, num_samples, num_features, key):
    weights = jr.normal(key, [num_samples * 2, 2 * num_features])
    return jnp.repeat(weights, dataset_shape // 2, axis=0)  # TODO generlise both to dims more than 2


class MO_RFF(AbstractKernel):
    r"""Computes an approximation of the kernel using Random Fourier Features.

    All stationary kernels are equivalent to the Fourier transform of a probability
    distribution. We call the corresponding distribution the spectral density. Using
    a finite number of basis functions, we can compute the spectral density using a
    Monte-Carlo approximation. This is done by sampling from the spectral density and
    computing the Fourier transform of the samples. The kernel is then approximated by
    the inner product of the Fourier transform of the samples with the Fourier
    transform of the data.

    The key reference for this implementation is the following papers:
    - 'Random Features for Large-Scale Kernel Machines' by Rahimi and Recht (2008).
    - 'On the Error of Random Fourier Features' by Sutherland and Schneider (2015).
    """

    compute_engine: BasisFunctionComputation

    def __init__(self,
                 kernel0: StationaryKernel,
                 kernel1: StationaryKernel,
                 num_basis_fns: int = 50,
                 frequencies: tp.Union[Float[Array, "M D"], None] = None,
                 compute_engine: BasisFunctionComputation = BasisFunctionComputation(),
                 key: KeyArray = jr.PRNGKey(0)):
        r"""Initialise the RFF kernel.

        Args:
            base_kernel (StationaryKernel): The base kernel to be approximated.
            num_basis_fns (int): The number of basis functions to use in the approximation.
            frequencies (Float[Array, "M D"] | None): The frequencies to use in the approximation.
                If None, the frequencies are sampled from the spectral density of the base
                kernel.
            compute_engine (BasisFunctionComputation): The computation engine to use for
                the basis function computation.
            key (KeyArray): The random key to use for sampling the frequencies.
        """
        self._check_valid_base_kernel(kernel0)
        self._check_valid_base_kernel(kernel1)
        self.kernel0 = kernel0
        self.kernel1 = kernel1
        self.num_basis_fns = num_basis_fns
        self.frequencies = frequencies
        self.compute_engine = compute_engine

        if self.frequencies is None:
            n_dims = self.kernel0.n_dims
            if n_dims is None:
                raise ValueError(
                    "Expected the number of dimensions to be specified for the base kernel. "
                    "Please specify the n_dims argument for the base kernel."
                )
            key0, key1 = jrandom.split(key)
            self.frequencies0 = Static(
                self.kernel0.spectral_density.sample(
                    seed=key0, sample_shape=(self.num_basis_fns, n_dims)
                )
            )
            self.frequencies1 = Static(
                self.kernel1.spectral_density.sample(
                    seed=key1, sample_shape=(self.num_basis_fns, n_dims)
                )
            )
        self.name = f"{self.kernel0.name} (RFF)"

    def __call__(self, x: Float[Array, "D 1"], y: Float[Array, "D 1"]) -> None:
        """Superfluous for RFFs."""
        raise RuntimeError("RFFs do not have a kernel function.")

    @staticmethod
    def _check_valid_base_kernel(kernel: AbstractKernel):
        r"""Verify that the base kernel is valid for RFF approximation.

        Args:
            kernel (AbstractKernel): The kernel to be checked.
        """
        if not isinstance(kernel, StationaryKernel):
            raise TypeError("RFF can only be applied to stationary kernels.")

        # check that the kernel has a spectral density
        _ = kernel.spectral_density

    def compute_features(self, x: Float[Array, "N D"]) -> Float[Array, "N L"]:
        r"""Compute the features for the inputs.

        Args:
            x: A $N \times D$ array of inputs.

        Returns:
            Float[Array, "N L"]: A $N \times L$ array of features where $L = 2M$.
        """
        z = jnp.array(x[:, -1], dtype=int)

        # achieve the correct value via 'switches' that are either 1 or 0
        k0_switch = ((z + 1) % 2)
        k1_switch = z

        part_1 = self.compute_features_otherfile(self.frequencies0, self.kernel0, x[:, :-1]) * jnp.sqrt(self.kernel0.variance.value / self.num_basis_fns)
        part_2 = self.compute_features_otherfile(self.frequencies1, self.kernel1, x[:, :-1]) * jnp.sqrt(self.kernel1.variance.value / self.num_basis_fns)
        return jnp.expand_dims(k0_switch, axis=-1) * part_1 + jnp.expand_dims(k1_switch, axis=-1) * part_2

    def compute_features_otherfile(self, kernel_frequencies, kernel: K, x: Float[Array, "N D"]) -> Float[Array, "N L"]:
        r"""Compute the features for the inputs.

        Args:
            kernel: the kernel function.
            x: the inputs to the kernel function of shape `(N, D)`.

        Returns:
            A matrix of shape $N \times L$ representing the random fourier features where $L = 2M$.
        """
        frequencies = kernel_frequencies.value
        scaling_factor = kernel.lengthscale.value
        z = jnp.matmul(x, (frequencies / scaling_factor).T)
        z = jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)
        return z

def adjust_dataset(x, y):
    # Change vectors x -> X = (x,z), and vectors y -> Y = (y,z) via the artificial z label
    def label_position(data):  # 2,20
        # introduce alternating z label
        n_points = len(data[0])
        label = jnp.tile(jnp.array([0.0, 1.0]), n_points)
        return jnp.vstack((jnp.repeat(data, repeats=2, axis=1), label)).T

    # change vectors y -> Y by reshaping the velocity measurements
    def stack_velocity(data):  # 2,20
        return data.T.flatten().reshape(-1, 1)

    def dataset_3d(pos, vel):
        return gpjax.Dataset(label_position(pos), stack_velocity(vel))

    # takes in dimension (number of data points, num features)

    return dataset_3d(jnp.swapaxes(x, 0, 1), jnp.swapaxes(y, 0, 1))

import gpjax
from gpjax.kernels.computations import DenseKernelComputation

class VelocityKernel(gpjax.kernels.stationary.StationaryKernel):  #TODO changed this from abstract kernel
    def __init__(
        self,
        # kernel0: gpjax.kernels.AbstractKernel = gpjax.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0),  # TODO the original with abstract kernels
        # kernel1: gpjax.kernels.AbstractKernel = gpjax.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0),
        kernel0: gpjax.kernels.stationary.StationaryKernel = gpjax.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0),
        kernel1: gpjax.kernels.stationary.StationaryKernel = gpjax.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0),
        # kernel1: gpjax.kernels.stationary.StationaryKernel = gpjax.kernels.Matern32(),
    ):
        self.kernel0 = kernel0
        self.kernel1 = kernel1
        super().__init__(n_dims=3, compute_engine=DenseKernelComputation())

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        # standard RBF-SE kernel is x and x' are on the same output, otherwise returns 0

        z = jnp.array(X[-1], dtype=int)
        zp = jnp.array(Xp[-1], dtype=int)

        # achieve the correct value via 'switches' that are either 1 or 0
        k0_switch = ((z + 1) % 2) * ((zp + 1) % 2)
        k1_switch = z * zp

        return k0_switch * self.kernel0(X, Xp) + k1_switch * self.kernel1(X, Xp)


def tests():
    import gpjax
    import jax.random as jrandom
    from jax import config

    def label_position(data):
        # introduce alternating z label
        n_points = len(data[0])
        label = jnp.tile(jnp.array([0.0, 1.0]), n_points)
        return jnp.vstack((jnp.repeat(data, repeats=2, axis=1), label)).T

    # change vectors y -> Y by reshaping the velocity measurements
    def stack_velocity(data):
        return data.T.flatten().reshape(-1, 1)

    def dataset_3d(pos, vel):
        return gpjax.Dataset(label_position(pos), stack_velocity(vel))

    main_key = jrandom.key(42)

    config.update("jax_enable_x64", True)

    mean_func = gpjax.mean_functions.Zero()

    kernel_mo = VelocityKernel()
    prior_mo = gpjax.gps.Prior(mean_function=mean_func,  kernel=kernel_mo)

    kernel = gpjax.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0)
    prior_ind = gpjax.gps.Prior(mean_function=mean_func, kernel=kernel)
    prior = [prior_ind for _ in range(2)]

    # so data is of shape x,2 and we copy for x and then we split y into indivudal gps, basically buffer the dataset

    num_data = 4
    train_data = gpjax.Dataset(jnp.arange(0, num_data, 1, dtype=jnp.float64).reshape(-1, 2), jnp.arange(num_data, num_data*2, 1, dtype=jnp.float64).reshape(-1, 2))
    train_data_mo = adjust_dataset(train_data.X, train_data.y)

    test_data = gpjax.Dataset(jnp.arange(num_data * 2, num_data * 3 - (num_data / 2), 1, dtype=jnp.float64).reshape(-1, 2), jnp.arange(num_data * 2, num_data * 3 - (num_data / 2), 1, dtype=jnp.float64).reshape(-1, 2))
    test_data_mo = adjust_dataset(test_data.X, test_data.y)

    new_vals = []
    key = main_key.copy()
    for gp_idx, ind_prior in enumerate(prior):
        data = gpjax.Dataset(X=train_data.X, y=jnp.expand_dims(train_data.y[:, gp_idx], axis=-1))
        likelihood = gpjax.likelihoods.Gaussian(data.n)
        posterior = ind_prior * likelihood
        # key, _key = jrandom.split(key)
        sample_func = posterior.sample_approx(num_samples=1, train_data=data, key=main_key, num_features=3)
        new_vals.append(sample_func(test_data.X))

    posterior_mo = prior_mo * gpjax.likelihoods.Gaussian(train_data_mo.n)
    sample_func_mo = adj_sample_approx(posterior_mo, num_samples=1, train_data=train_data_mo, key=main_key, num_features=3)
    new_vals_mo = sample_func_mo(test_data_mo.X)
    # latent = posterior_mo.predict(test_data_mo.X, train_data=train_data_mo)
    # latent_mean = latent.mean()

    # new_data = dataset_3d(test_data.X, latent_mean)

    print(jnp.concatenate((new_vals[0], new_vals[1]), axis=-1))
    print(jnp.swapaxes(new_vals_mo, 0, 1))

    return


if __name__ == "__main__":
    tests()