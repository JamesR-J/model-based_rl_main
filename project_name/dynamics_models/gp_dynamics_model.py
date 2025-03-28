import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import struct
from flax.training import train_state
import optax
from typing import List, Tuple, Dict, Optional, NamedTuple, Any
from functools import partial
import GPJax_AScannell as gpjaxas
import optax

from project_name.dynamics_models import DynamicsModelBase
from GPJax_AScannell.gpjax.config import default_jitter
import jax.scipy as jsp
from jaxtyping import Float, install_import_hook

with install_import_hook("gpjax", "beartype.beartype"):
    import logging
    logging.getLogger('gpjax').setLevel(logging.WARNING)
    import gpjax


class MOGP(DynamicsModelBase):
    def __init__(self, env, env_params, config, agent_config, key):
        super().__init__(env, env_params, config, agent_config, key)
        # TODO ls below is for cartpole
        # ls = jnp.array([[80430.37, 4.40, 116218.45, 108521.27, 103427.47], [290.01, 318.22, 0.39, 1.57, 33.17],
        #      [1063051.24, 1135236.37, 1239430.67, 25.09, 1176016.11], [331.70, 373.98, 0.32, 1.88, 39.83]], dtype=jnp.float64)
        self.ls = jnp.array([[2.27, 7.73, 138.94], [0.84, 288.15, 11.05]],
                       dtype=jnp.float64)
        alpha = jnp.array([0.26, 2.32, 11.59, 3.01], dtype=jnp.float64)  # for periodic kernels
        self.sigma = 0.01

        self.kernel = gpjaxas.kernels.SeparateIndependent([gpjaxas.kernels.SquaredExponential(lengthscales=self.ls[idx], variance=self.sigma) for idx in range(self.obs_dim)])
        self.likelihood = gpjaxas.likelihoods.Gaussian(variance=3.0)
        self.mean_function = gpjaxas.mean_functions.Zero(output_dim=self.obs_dim)
        self.gp = gpjaxas.models.GPR(self.kernel, self.likelihood, self.mean_function, num_latent_gps=self.obs_dim)

    def create_train_state(self, init_data_x, init_data_y, key):
        params = self.gp.get_params()
        return params

    def create_sep_params(self, params):
        kernels = params["kernel"]
        lengthscales = jnp.stack([k['lengthscales'] for k in kernels])
        variances = jnp.stack([k['variance'] for k in kernels])
        likelihood_variances = jnp.tile(params["likelihood"]["variance"], (self.obs_dim, 1))

        params["kernel"] = {"lengthscales": lengthscales, "variance": variances}
        params["likelihood"] = {"variance": likelihood_variances}

        return params

    @partial(jax.jit, static_argnums=(0,))
    def optimise(self, train_data, params):
        # # transforms = self.gp.get_transforms()
        # # constrain_params = gpjaxas.parameters.build_constrain_params(transforms)
        # # params = constrain_params(params)
        #
        # def create_sep_transform(params):
        #     params["kernel"] = {"lengthscales": params["kernel"][0]["lengthscales"],
        #                         "variance": params["kernel"][0]["variance"]}
        #     return params

        sep_params = self.create_sep_params(params)
        # sep_transforms = create_sep_transform(transforms)
        # sep_constrain_params = gpjaxas.parameters.build_constrain_params(sep_transforms)
        # sep_params = sep_constrain_params(sep_params)

        objective = lambda p, d: -self.gp.multi_output_log_marginal_likelihood(p, d)
        tx = optax.adam(self.agent_config.GP_LR)

        def optimise_single_gp(data_x, data_y, init_params, idx):
            # Create optimiser state
            opt_state = tx.init(init_params)
            train_data = gpjax.Dataset(data_x, jnp.expand_dims(data_y, axis=-1))

            # Optimisation step.
            def _step_fn(carry, unused):
                params, opt_state = carry
                # params = sep_constrain_params(params)

                loss_val, loss_gradient = jax.value_and_grad(objective)(params, train_data)
                updates, opt_state = tx.update(loss_gradient, opt_state, params)
                params = optax.apply_updates(params, updates)

                carry = params, opt_state
                return carry, loss_val

            # Optimisation loop.
            (end_params, _), history = jax.lax.scan(jax.jit(_step_fn), (init_params, opt_state), None, self.agent_config.TRAIN_GP_NUM_ITERS)

            return end_params, history

        indices = jnp.linspace(0, 1, 2, dtype=jnp.int64)
        all_dim_params, loss_vals = jax.vmap(optimise_single_gp, in_axes=(None, 1, 0, 0))(train_data.X, train_data.y, sep_params, indices)

        return all_dim_params, loss_vals

    def pretrain_params(self, init_data_x, init_data_y, pretrain_data_x, pretrain_data_y, key):
        params = self.create_train_state(init_data_x, init_data_y, key)
        key, _key = jrandom.split(key)
        lengthscales = jrandom.uniform(_key, minval=0.0, maxval=100, shape=(self.agent_config.PRETRAIN_RESTARTS, self.obs_dim, self.obs_dim + self.action_dim))
        key, _key = jrandom.split(key)
        variances = jrandom.uniform(_key, minval=0.001, maxval=10.0, shape=(self.agent_config.PRETRAIN_RESTARTS, self.obs_dim, 1))

        def _batch_gp_training(data_x, data_y, params, lengthscales, variances):
            kernel = gpjaxas.kernels.SeparateIndependent(
                [gpjaxas.kernels.SquaredExponential(lengthscales=lengthscales[idx], variance=variances[idx]) for idx in
                 range(self.obs_dim)])
            # TODO a dodgy fix for now
            kernel_params = kernel.get_params()

            new_params = {"kernel": kernel_params, "likelihood": params["likelihood"], "mean_function": params["mean_function"]}

            train_data = gpjax.Dataset(data_x, data_y)

            return self.optimise(train_data, new_params)

        new_params, loss_vals = jax.vmap(_batch_gp_training, in_axes=(None, None, None, 0, 0))(pretrain_data_x, pretrain_data_y, params, lengthscales, variances)
        last_loss_vals = loss_vals[:, :, -1]
        best_idx = jnp.argmin(last_loss_vals, axis=0)
        n_indices = jnp.arange(last_loss_vals.shape[1])
        best_params = jax.tree_util.tree_map(lambda x: x[best_idx, n_indices, ...], new_params)
        kernel = gpjaxas.kernels.SeparateIndependent(
            [gpjaxas.kernels.SquaredExponential(lengthscales=best_params["kernel"]["lengthscales"][idx], variance=best_params["kernel"]["variance"][idx]) for idx in
             range(self.obs_dim)])
        # TODO dodgy fix for now assuming the kernels are the same

        kernel_params = kernel.get_params()
        params["kernel"] = kernel_params

        # TODO sort this as it also trains the likelihood variance yet only keeps the optimal kernel params

        return params

    # @partial(jax.jit, static_argnums=(0,))
    def get_post_mu_cov(self, XNew, params, train_data, full_cov=False):  # TODO if no data then return the prior mu and var
        mu, std = self.gp.predict_f(params, XNew, train_data, full_cov=full_cov)
        return mu, std

    # @partial(jax.jit, static_argnums=(0,))
    def get_post_mu_cov_samples(self, XNew, params, train_data, key, full_cov=False):
        samples = self.gp.predict_f_samples(params, key, XNew, train_data, num_samples=1, full_cov=full_cov)
        return samples[0] # TODO works on the first sample as only one required for now

    @partial(jax.jit, static_argnums=(0,))
    def predict_on_noisy_inputs(self, m, s, train_state, train_data):
        iK, beta = self.calculate_factorizations(train_state, train_data)
        return self.predict_given_factorizations(m, s, iK, beta, train_state, train_data)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_factorizations(self, train_state, train_data):
        X = train_data.X
        Y = train_data.y
        Kmm = self.gp.kernel(train_state["kernel"], X, X) # [..., M, M]
        batched_eye = jnp.expand_dims(jnp.eye(jnp.shape(X)[0]), axis=0).repeat(self.obs_dim, axis=0)
        noise = train_state["likelihood"]["variance"][:, None, None].repeat(self.obs_dim, axis=0)
        L = jsp.linalg.cho_factor(Kmm + noise * batched_eye, lower=True)
        iK = jsp.linalg.cho_solve(L, batched_eye)
        Y_ = jnp.transpose(Y)[:, :, None]
        beta = jsp.linalg.cho_solve(L, Y_)[:, :, 0]
        return iK, beta

    @partial(jax.jit, static_argnums=(0,))
    def predict_given_factorizations(self, m, s, iK, beta, train_state, train_data):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """
        sep_params = self.create_sep_params(train_state)
        variance = sep_params["kernel"]["variance"]
        lengthscales = sep_params["kernel"]["lengthscales"]

        s = jnp.tile(s[None, None, :, :], [self.obs_dim, self.obs_dim, 1, 1])
        inp = jnp.tile(self.centralised_input(train_data.X, m)[None, :, :], [self.obs_dim, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = jax.vmap(lambda x: jnp.diag(x, k=0))(1 / lengthscales)
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + jnp.eye(self.obs_dim + self.action_dim)

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = jnp.transpose(jnp.linalg.solve(B, jnp.transpose(iN, axes=(0, 2, 1))), axes=(0, 2, 1))

        lb = jnp.exp(-0.5 * jnp.sum(iN * t, -1)) * beta
        tiL = t @ iL
        c = variance / jnp.sqrt(jnp.linalg.det(B))

        M = (jnp.sum(lb, -1) * c)[:, None]
        V = (jnp.transpose(tiL, axes=(0, 2, 1)) @ lb[:, :, None])[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        z = jax.vmap(jax.vmap(lambda x: jnp.diag(x, k=0)))(1.0 / jnp.square(lengthscales[None, :, :]) + 1.0 / jnp.square(lengthscales[:, None, :]))

        R = (s @ z) + jnp.eye(self.obs_dim + self.action_dim)

        X = inp[None, :, :, :] / jnp.square(lengthscales[:, None, None, :])
        X2 = -inp[:, None, :, :] / jnp.square(lengthscales[None, :, None, :])
        Q = 0.5 * jnp.linalg.solve(R, s)
        maha = (X - X2) @ Q @ jnp.transpose(X - X2, axes=(0, 1, 3, 2))

        k = jnp.log(variance)[:, None] - 0.5 * jnp.sum(jnp.square(iN), -1)
        L = jnp.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = (
            jnp.tile(beta[:, None, None, :], [1, self.obs_dim, 1, 1])
            @ L
            @ jnp.tile(beta[None, :, :, None], [self.obs_dim, 1, 1, 1])
        )[:, :, 0, 0]

        diagL = jnp.transpose(jax.vmap(jax.vmap(lambda x: jnp.diag(x, k=0)))(jnp.transpose(L)))
        S = S - jnp.diag(jnp.sum(jnp.multiply(iK, diagL), [1, 2]))
        S = S / jnp.sqrt(jnp.linalg.det(R))
        S = S + jnp.diag(variance)
        S = S - M @ jnp.transpose(M)

        return jnp.transpose(M), S, jnp.transpose(V)

    @partial(jax.jit, static_argnums=(0,))
    def centralised_input(self, X, m):
        return X - m
