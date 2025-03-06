import jax
import jax.numpy as jnp
import jax.random as jrandom
import gpjax
from flax import struct
from flax.training import train_state
import optax
from typing import List, Tuple, Dict, Optional, NamedTuple, Any
from functools import partial
import GPJax_AScannell as gpjaxas
import optax
import gpjax


class MOGP:
    def __init__(self, env, env_params, config, utils, key):
        self.config = config
        # self.agent_config = get_MPC_config()  # TODO create a GP config thing
        self.env = env
        self.env_params = env_params

        self.obs_dim = len(self.env.observation_space(self.env_params).low)
        self.action_dim = self.env.action_space().shape[0]

        self.input_dim = self.obs_dim + self.action_dim

        num_latent_gps = self.obs_dim

        # TODO ls below is for cartpole
        # ls = jnp.array([[80430.37, 4.40, 116218.45, 108521.27, 103427.47], [290.01, 318.22, 0.39, 1.57, 33.17],
        #      [1063051.24, 1135236.37, 1239430.67, 25.09, 1176016.11], [331.70, 373.98, 0.32, 1.88, 39.83]], dtype=jnp.float64)
        ls = jnp.array([[2.27, 7.73, 138.94], [0.84, 288.15, 11.05]],
                       dtype=jnp.float64)
        alpha = jnp.array([0.26, 2.32, 11.59, 3.01], dtype=jnp.float64)  # TODO what is alpha?
        sigma = 0.01  # TODO right?

        self.kernel = gpjaxas.kernels.SeparateIndependent([gpjaxas.kernels.SquaredExponential(lengthscales=ls[idx], variance=sigma) for idx in range(self.obs_dim)])
        self.likelihood = gpjaxas.likelihoods.Gaussian(variance=3.0)
        self.mean_function = gpjaxas.mean_functions.Zero(output_dim=self.action_dim)
        self.gp = gpjaxas.models.GPR(self.kernel, self.likelihood, self.mean_function, num_latent_gps=num_latent_gps)

    def create_train_state(self, init_data_x, init_data_y):
        params = self.gp.get_params()
        params["train_data_x"] = init_data_x
        params["train_data_y"] = init_data_y
        return params

    def get_post_mu_cov(self, XNew, params, train_data=None, full_cov=False):  # TODO if no data then return the prior mu and var
        mu, std = self.gp.predict_f(params, XNew, train_data=train_data, full_cov=full_cov)
        return mu, std

    def return_posterior_samples(self, XNew, params, key):
        samples = self.gp.predict_f_samples(params, key, XNew)
        return samples


####### underneath are tests, above is trying with aidan scannell package


class GaussianProcessParameters(NamedTuple):
    log_error_stddev: jnp.ndarray
    kernel_params: NamedTuple

class GaussianProcessState(NamedTuple):
    prior_frequency: jnp.ndarray
    prior_phase: jnp.ndarray
    prior_weights: jnp.ndarray
    cholesky: jnp.ndarray


class GPDynamicsModel():  # TODO create a general baseline for one
    def __init__(self, env, env_params, config, utils, key):
        self.config = config
        # self.agent_config = get_MPC_config()  # TODO create a GP config thing
        self.env = env
        self.env_params = env_params

        self.obs_dim = len(self.env.observation_space(self.env_params).low)
        self.action_dim = self.env.action_space().shape[0]

        self.mean = gpjax.mean_functions.Zero()
        self.kernel = gpjax.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0)  # TODO make this more general
        self.prior = gpjax.gps.Prior(mean_function=self.mean, kernel=self.kernel)

        self.gp_models = [self.prior for _ in range(self.obs_dim)]

    def get_post_mu_cov(self, x_data):
        mu = "STR"
        std = "STR"
        return mu, std


# Reusing the GP implementation from before
class GPKernel(struct.PyTreeNode):
    """Base class for Gaussian Process kernels."""

    def __call__(self, x1, x2):
        """Compute kernel matrix between inputs x1 and x2."""
        raise NotImplementedError

    def spectral_weights(self, frequency):
        """Compute spectral weights for random feature approximation."""
        raise NotImplementedError

    def standard_spectral_measure(self, key, num_samples):
        """Sample from the spectral measure of the kernel."""
        raise NotImplementedError


class RBFKernel(GPKernel):
    """Radial Basis Function (RBF) kernel, also known as squared exponential."""
    length_scale: float = 1.0
    signal_variance: float = 1.0

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x1, x2):
        """Compute RBF kernel matrix between inputs x1 and x2."""
        # Add batch dimensions if needed
        x1 = jnp.atleast_1d(x1)
        x2 = jnp.atleast_1d(x2)

        # Reshape for broadcasting
        x1_expanded = jnp.expand_dims(x1, axis=-2)  # [N, 1, D]
        x2_expanded = jnp.expand_dims(x2, axis=-3)  # [1, M, D]

        # Compute squared distances
        sq_dist = jnp.sum(jnp.square(x1_expanded - x2_expanded), axis=-1)

        # Compute kernel values
        return self.signal_variance * jnp.exp(-0.5 * sq_dist / (self.length_scale ** 2))

    @partial(jax.jit, static_argnums=(0,))
    def spectral_weights(self, frequency):
        """Compute spectral weights for RBF kernel."""
        # For RBF kernel, the spectral measure is Gaussian
        outer_weights = self.signal_variance * jnp.ones_like(frequency[..., 0])
        inner_weights = self.length_scale * jnp.ones_like(frequency[..., 0:1])
        return outer_weights, inner_weights

    @partial(jax.jit, static_argnums=(0,))
    def standard_spectral_measure(self, key, num_samples):
        """Sample from the spectral measure of RBF kernel."""
        # For RBF kernel, the spectral measure is Gaussian
        # The spectral density is (2π)^(-d/2) * exp(-ω²/2)
        return jrandom.normal(key, (num_samples,))


class GaussianProcess:
    def __init__(self, env, env_params, config, utils, key):
        self.config = config
        # self.agent_config = get_MPC_config()  # TODO create a GP config thing
        self.env = env
        self.env_params = env_params

        self.obs_dim = len(self.env.observation_space(self.env_params).low)
        self.action_dim = self.env.action_space().shape[0]

        self.kernel = RBFKernel()

    def create_train_state(self):
        return (TrainState.create(apply_fn=self.network.apply,
                                  params=self.network_params,
                                  tx=self.tx),
                MemoryState(hstate=jnp.zeros((self.config.NUM_ENVS, 1)),
                            extras={
                                "values": jnp.zeros((self.config.NUM_ENVS, 1)),
                                "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
                            })
                )


    @partial(jax.jit, static_argnums=(0,))
    def predict(self, x_train, y_train, x_test, diag=False):
        """
        Make GP predictions at test points given training data.

        Args:
            x_train: Training inputs [N, D]
            y_train: Training targets [N]
            x_test: Test inputs [M, D]
            diag: If True, return only diagonal of predictive covariance

        Returns:
            mean: Predictive mean [M]
            var: Predictive variance [M] or covariance [M, M]
        """
        # Compute kernel matrices
        k_xx = self.kernel(x_train, x_train)
        k_xx_inv = jnp.linalg.inv(k_xx + self.noise_variance * jnp.eye(x_train.shape[0]))
        k_xs = self.kernel(x_train, x_test)
        k_ss = self.kernel(x_test, x_test)

        # Compute predictive mean
        mean = k_xs.T @ k_xx_inv @ y_train

        # Compute predictive variance/covariance
        if diag:
            # Efficient diagonal-only computation
            var_diag = jnp.diag(k_ss) - jnp.sum(k_xs.T * (k_xx_inv @ k_xs), axis=1)
            return mean, var_diag
        else:
            # Full covariance matrix
            cov = k_ss - k_xs.T @ k_xx_inv @ k_xs
            return mean, cov


class GPState(train_state.TrainState):
    """Train state for Gaussian Process."""
    gp: GaussianProcess
    x_train: jnp.ndarray = None
    y_train: jnp.ndarray = None

    @classmethod
    def create(cls, *, apply_fn=None, params=None, tx=None, gp, x_train=None, y_train=None, **kwargs):
        """Factory method for creating a GPState."""
        state = super().create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            **kwargs
        )
        return cls(
            **state.__dict__,
            gp=gp,
            x_train=x_train,
            y_train=y_train
        )

    @partial(jax.jit, static_argnums=(0,))
    def log_marginal_likelihood(self):
        """Compute log marginal likelihood of GP given current data."""
        if self.x_train is None or self.y_train is None:
            return jnp.array(0.0)

        n = self.x_train.shape[0]
        K = self.gp.kernel(self.x_train, self.x_train)
        K_y = K + self.gp.noise_variance * jnp.eye(n)

        # Stable computation using Cholesky
        L = jnp.linalg.cholesky(K_y)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.y_train))

        # Compute log marginal likelihood
        lml = -0.5 * jnp.dot(self.y_train, alpha)
        lml -= jnp.sum(jnp.log(jnp.diag(L)))
        lml -= 0.5 * n * jnp.log(2 * jnp.pi)

        return lml

    @partial(jax.jit, static_argnums=(0,))
    def update_data(self, x_new, y_new):
        """Update training data with new observations."""
        if self.x_train is None:
            x_train = x_new
            y_train = y_new
        else:
            x_train = jnp.concatenate([self.x_train, x_new])
            y_train = jnp.concatenate([self.y_train, y_new])

        return self.replace(x_train=x_train, y_train=y_train)

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, x_test, diag=True):
        """Make predictions using current GP model and training data."""
        if self.x_train is None or self.y_train is None:
            # Return prior if no data
            mean = jnp.zeros(x_test.shape[0])
            if diag:
                var = jnp.ones(x_test.shape[0]) * self.gp.kernel.signal_variance
            else:
                var = self.gp.kernel(x_test, x_test)
            return mean, var

        return self.gp.predict(self.x_train, self.y_train, x_test, diag=diag)


# Create a GP optimizer
def create_gp_optimizer(learning_rate=0.01):
    """Create an optimizer for GP hyperparameters."""
    return optax.adam(learning_rate)


# Init function that can be easily vmapped
def init_gp(key, length_scale=1.0, signal_variance=1.0, noise_variance=0.1):
    """Initialize a GP state that can be easily vmapped."""
    kernel = RBFKernel(length_scale=length_scale, signal_variance=signal_variance)
    gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)

    # Create initial parameters dict for optimizer
    params = {
        'kernel': {
            'length_scale': length_scale,
            'signal_variance': signal_variance
        },
        'noise_variance': noise_variance
    }

    # Create optimizer
    tx = create_gp_optimizer()

    # Create train state
    state = GPState.create(
        apply_fn=None,  # Not needed for GP
        params=None,  # Not needed in standard form
        tx=tx,
        gp=gp,
        opt_state=tx.init(params)
    )

    return state


# ============= MULTI-OUTPUT GP IMPLEMENTATION =============== #

class MultiOutputGP(struct.PyTreeNode):
    """Multi-output Gaussian Process using independent GPs for each output dimension."""
    output_gps: List[GPState]
    output_dims: int

    @classmethod
    def create(cls, key, output_dims, **gp_kwargs):
        """Create a multi-output GP with independent GPs for each output."""
        # Split the key for each output GP
        keys = jax.random.split(key, output_dims)

        # Create a GP for each output dimension
        output_gps = jax.vmap(init_gp)(keys, **gp_kwargs)

        return cls(output_gps=output_gps, output_dims=output_dims)

    def update_data(self, x, y):
        """
        Update all GPs with new data.

        Args:
            x: Input features [N, D]
            y: Output values [N, output_dims]

        Returns:
            Updated MultiOutputGP
        """
        # Update each GP with its corresponding output dimension
        updated_gps = jax.vmap(lambda gp, y_col: gp.update_data(x, y_col))(
            self.output_gps, y.T
        )

        return self.replace(output_gps=updated_gps)

    def predict(self, x_test, diag=True):
        """
        Make predictions with all output GPs.

        Args:
            x_test: Test points [M, D]
            diag: If True, return only diagonal of predictive covariance

        Returns:
            means: Predictive means [M, output_dims]
            vars: Predictive variances [M, output_dims] or covariances [output_dims, M, M]
        """
        # Vmap the predict function across all output GPs
        means, vars = jax.vmap(lambda gp: gp.predict(x_test, diag=diag))(self.output_gps)

        # Transpose means to have shape [M, output_dims]
        means = means.T

        if diag:
            # Transpose variances to have shape [M, output_dims]
            vars = vars.T

        return means, vars

    def log_marginal_likelihood(self):
        """Compute total log marginal likelihood across all outputs."""
        # Sum log marginal likelihoods from all output GPs
        lml = jax.vmap(lambda gp: gp.log_marginal_likelihood())(self.output_gps)
        return jnp.sum(lml)


def train_step_mogp(mogp, batch, learning_rate=0.01):
    """Single training step for multi-output GP hyperparameters."""
    x_batch, y_batch = batch

    # Define per-output training step
    def train_single_output(gp_state, y_output):
        """Train a single output GP."""

        def loss_fn(params):
            # Update GP with new parameters
            kernel = RBFKernel(**params['kernel'])
            noise_variance = params['noise_variance']
            gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)

            # Create temporary state with updated GP
            temp_state = gp_state.replace(gp=gp)

            # Compute negative log marginal likelihood
            return -temp_state.log_marginal_likelihood()

        # Extract current parameters
        params = {
            'kernel': {
                'length_scale': gp_state.gp.kernel.length_scale,
                'signal_variance': gp_state.gp.kernel.signal_variance
            },
            'noise_variance': gp_state.gp.noise_variance
        }

        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Create optimizer if needed
        tx = create_gp_optimizer(learning_rate)
        opt_state = gp_state.opt_state if gp_state.opt_state is not None else tx.init(params)

        # Update parameters
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Create new GP with updated parameters
        new_kernel = RBFKernel(**new_params['kernel'])
        new_gp = GaussianProcess(kernel=new_kernel, noise_variance=new_params['noise_variance'])

        # Update state
        new_state = gp_state.replace(
            gp=new_gp,
            opt_state=new_opt_state
        )

        return new_state, loss

    # Train each output GP
    new_gp_states, losses = jax.vmap(train_single_output)(
        mogp.output_gps, y_batch.T
    )

    # Create updated multi-output GP
    new_mogp = mogp.replace(output_gps=new_gp_states)

    # Return total loss across all outputs
    total_loss = jnp.sum(losses)

    return new_mogp, total_loss


# Example usage of multi-output GP
def example_mogp():
    # Initialize random key
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Create a simple dataset with 2 output dimensions
    n_samples = 50
    x_train = jnp.linspace(-5, 5, n_samples).reshape(-1, 1)

    # Create two correlated outputs
    y1 = jnp.sin(x_train[:, 0]) + 0.1 * jax.random.normal(subkey1, (n_samples,))
    y2 = jnp.cos(x_train[:, 0]) + 0.1 * jax.random.normal(subkey2, (n_samples,))
    y_train = jnp.stack([y1, y2], axis=1)  # Shape [n_samples, 2]

    # Create a multi-output GP with 2 output dimensions
    key, subkey = jax.random.split(key)
    mogp = MultiOutputGP.create(
        subkey,
        output_dims=2,
        length_scale=jnp.ones(2),  # One length scale per output
        signal_variance=jnp.ones(2),  # One signal variance per output
        noise_variance=0.1 * jnp.ones(2)  # One noise variance per output
    )

    # Update with training data
    mogp = mogp.update_data(x_train, y_train)

    # Make predictions
    x_test = jnp.linspace(-7, 7, 100).reshape(-1, 1)
    means, vars = mogp.predict(x_test)

    # Train the multi-output GP
    for i in range(100):
        mogp, loss = train_step_mogp(mogp, (x_train, y_train))
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss}")

    # Make predictions after training
    means_trained, vars_trained = mogp.predict(x_test)

    return {
        "initial_predictions": (means, vars),
        "trained_predictions": (means_trained, vars_trained)
    }


# More advanced: LMC (Linear Model of Coregionalization)
class LMCMultiOutputGP(struct.PyTreeNode):
    """
    Multi-output GP using Linear Model of Coregionalization.

    This model uses a shared set of latent GPs and combines them
    with a linear transformation matrix to model correlations between outputs.
    """
    latent_gps: List[GPState]
    mixing_matrix: jnp.ndarray  # Shape [output_dims, num_latent_gps]
    output_dims: int
    num_latent_gps: int

    @classmethod
    def create(cls, key, output_dims, num_latent_gps=None, **gp_kwargs):
        """
        Create a LMC multi-output GP.

        Args:
            key: JAX random key
            output_dims: Number of output dimensions
            num_latent_gps: Number of latent GPs (defaults to output_dims)
            **gp_kwargs: Arguments for GP initialization
        """
        if num_latent_gps is None:
            num_latent_gps = output_dims

        # Split keys for latent GPs and mixing matrix
        key, subkey_gps, subkey_mix = jax.random.split(key, 3)

        # Create latent GPs
        latent_keys = jax.random.split(subkey_gps, num_latent_gps)
        latent_gps = jax.vmap(init_gp)(latent_keys, **gp_kwargs)

        # Initialize mixing matrix
        mixing_matrix = jax.random.normal(subkey_mix, (output_dims, num_latent_gps))

        return cls(
            latent_gps=latent_gps,
            mixing_matrix=mixing_matrix,
            output_dims=output_dims,
            num_latent_gps=num_latent_gps
        )

    def update_data(self, x, y):
        """
        Update the model with new data.

        For LMC, this is more complex as we need to infer the latent functions.
        Here, we'll use a simple approach of updating each latent GP with a
        weighted combination of the outputs.

        Args:
            x: Input features [N, D]
            y: Output values [N, output_dims]

        Returns:
            Updated LMCMultiOutputGP
        """
        # For each latent GP, create a weighted combination of outputs
        # This is a simplified approach - in practice, you might want to use
        # more sophisticated inference methods

        # Normalize mixing matrix for weighted combinations
        weights = self.mixing_matrix / (jnp.sum(jnp.abs(self.mixing_matrix), axis=0) + 1e-6)

        # Create weighted combinations of outputs for each latent GP
        latent_y = y @ weights  # Shape [N, num_latent_gps]

        # Update each latent GP with its corresponding pseudo-observations
        updated_gps = jax.vmap(lambda gp, y_latent: gp.update_data(x, y_latent))(
            self.latent_gps, latent_y.T
        )

        return self.replace(latent_gps=updated_gps)

    def predict(self, x_test, diag=True):
        """
        Make predictions with the LMC model.

        Args:
            x_test: Test points [M, D]
            diag: If True, return only diagonal of predictive covariance

        Returns:
            means: Predictive means [M, output_dims]
            vars: Predictive variances [M, output_dims] or covariances
        """
        # Get predictions from all latent GPs
        latent_means, latent_vars = jax.vmap(lambda gp: gp.predict(x_test, diag=diag))(self.latent_gps)

        # Combine latent predictions using the mixing matrix
        # Shape transformation: [num_latent_gps, M] -> [M, output_dims]
        output_means = jnp.einsum('lm,ol->mo', latent_means, self.mixing_matrix)

        if diag:
            # For diagonal covariance, we need to account for correlations in the outputs
            # This is a simplified approach that assumes independence between latent GPs
            output_vars = jnp.einsum('lm,ol,ol->mo', latent_vars, self.mixing_matrix, self.mixing_matrix)
            return output_means, output_vars
        else:
            # For full covariance, we would need to implement a more complex calculation
            # This is a placeholder for full covariance calculation
            raise NotImplementedError("Full covariance prediction for LMC not yet implemented")

    def log_marginal_likelihood(self):
        """
        Compute an approximation to the log marginal likelihood.

        Note: This is just a proxy based on the latent GPs' likelihoods.
        A proper implementation would compute the full joint likelihood.
        """
        # Sum log marginal likelihoods from all latent GPs
        lml = jax.vmap(lambda gp: gp.log_marginal_likelihood())(self.latent_gps)
        return jnp.sum(lml)


# Example usage with LMC
def example_lmc_mogp():
    # Initialize random key
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Create a simple dataset with 3 output dimensions and strong correlations
    n_samples = 50
    x_train = jnp.linspace(-5, 5, n_samples).reshape(-1, 1)

    # Create three correlated outputs from two latent functions
    f1 = jnp.sin(x_train[:, 0])
    f2 = jnp.cos(x_train[:, 0])

    # Output 1 is mainly f1
    # Output 2 is a mix of f1 and f2
    # Output 3 is mainly f2
    y1 = f1 + 0.1 * jax.random.normal(subkey1, (n_samples,))
    y2 = 0.7 * f1 + 0.7 * f2 + 0.1 * jax.random.normal(subkey1, (n_samples,))
    y3 = f2 + 0.1 * jax.random.normal(subkey2, (n_samples,))

    y_train = jnp.stack([y1, y2, y3], axis=1)  # Shape [n_samples, 3]

    # Create an LMC multi-output GP with 3 output dimensions and 2 latent GPs
    key, subkey = jax.random.split(key)
    lmc_mogp = LMCMultiOutputGP.create(
        subkey,
        output_dims=3,
        num_latent_gps=2,
        length_scale=jnp.ones(2),
        signal_variance=jnp.ones(2),
        noise_variance=0.1 * jnp.ones(2)
    )

    # Initialize with true mixing matrix for demonstration
    true_mixing = jnp.array([
        [1.0, 0.0],  # Output 1 depends on latent GP 1
        [0.7, 0.7],  # Output 2 depends on both
        [0.0, 1.0]  # Output 3 depends on latent GP 2
    ])

    lmc_mogp = lmc_mogp.replace(mixing_matrix=true_mixing)

    # Update with training data
    lmc_mogp = lmc_mogp.update_data(x_train, y_train)

    # Make predictions
    x_test = jnp.linspace(-7, 7, 100).reshape(-1, 1)
    means, vars = lmc_mogp.predict(x_test)

    return {
        "predictions": (means, vars),
        "mixing_matrix": lmc_mogp.mixing_matrix
    }