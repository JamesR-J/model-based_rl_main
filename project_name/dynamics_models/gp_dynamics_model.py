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

# import gpjax
from project_name.dynamics_models import DynamicsModelBase
from GPJax_AScannell.gpjax.config import default_jitter
import jax.scipy as jsp


class MOGP(DynamicsModelBase):
    def __init__(self, env, env_params, config, agent_config, key):
        super().__init__(env, env_params, config, agent_config, key)
        num_latent_gps = self.obs_dim

        # TODO ls below is for cartpole
        # ls = jnp.array([[80430.37, 4.40, 116218.45, 108521.27, 103427.47], [290.01, 318.22, 0.39, 1.57, 33.17],
        #      [1063051.24, 1135236.37, 1239430.67, 25.09, 1176016.11], [331.70, 373.98, 0.32, 1.88, 39.83]], dtype=jnp.float64)
        self.ls = jnp.array([[2.27, 7.73, 138.94], [0.84, 288.15, 11.05]],
                       dtype=jnp.float64)
        alpha = jnp.array([0.26, 2.32, 11.59, 3.01], dtype=jnp.float64)  # TODO what is alpha? is this periodic?
        self.sigma = 0.01

        self.kernel = gpjaxas.kernels.SeparateIndependent([gpjaxas.kernels.SquaredExponential(lengthscales=self.ls[idx], variance=self.sigma) for idx in range(self.obs_dim)])
        self.likelihood = gpjaxas.likelihoods.Gaussian(variance=3.0)
        self.mean_function = gpjaxas.mean_functions.Zero(output_dim=self.obs_dim)  # it was action_dim
        self.gp = gpjaxas.models.GPR(self.kernel, self.likelihood, self.mean_function, num_latent_gps=num_latent_gps)

    def create_train_state(self, init_data_x, init_data_y, key):
        params = self.gp.get_params()
        params["train_data_x"] = init_data_x
        params["train_data_y"] = init_data_y
        return params

    def optimise(self, data_x, data_y, params):
        transforms = self.gp.get_transforms()
        constrain_params = gpjaxas.parameters.build_constrain_params(transforms)
        params = constrain_params(params)

        def create_sep_params(params):
            kernels = params["kernel"]
            lengthscales = jnp.stack([k['lengthscales'] for k in kernels])
            variances = jnp.expand_dims(jnp.stack([k['variance'] for k in kernels]), axis=-1)
            likelihood_variances = jnp.tile(params["likelihood"]["variance"], (self.obs_dim, 1))

            params["kernel"] = {"lengthscales": lengthscales, "variance": variances}
            params["likelihood"] = {"variance": likelihood_variances}

            return params

        def create_sep_transform(params):
            params["kernel"] = {"lengthscales": params["kernel"][0]["lengthscales"],
                                "variance": params["kernel"][0]["variance"]}
            return params

        sep_params = create_sep_params(params)
        sep_transforms = create_sep_transform(transforms)
        sep_constrain_params = gpjaxas.parameters.build_constrain_params(sep_transforms)
        sep_params = sep_constrain_params(sep_params)

        objective = lambda p, d: -self.gp.multi_output_log_marginal_likelihood(p, d)
        tx = optax.adam(self.agent_config.GP_LR)

        def optimise_single_gp(data_x, data_y, ind_params, idx):
            # Create optimiser state
            opt_state = tx.init(ind_params)

            train_data = (data_x, jnp.expand_dims(data_y, axis=-1))

            # Optimisation step.
            def _step_fn(carry, unused):
                params, opt_state = carry
                params = sep_constrain_params(params)

                loss_val, loss_gradient = jax.value_and_grad(objective)(params, train_data)
                updates, opt_state = tx.update(loss_gradient, opt_state, params)
                params = optax.apply_updates(params, updates)

                carry = params, opt_state
                return carry, loss_val

            # Optimisation loop.
            (ind_params, _), history = jax.lax.scan(jax.jit(_step_fn), (ind_params, opt_state), None, self.agent_config.PRETRAIN_GP_NUM_ITERS)

            return ind_params, history

        indices = jnp.linspace(0, 1, 2, dtype=jnp.int64)
        all_params, loss_vals = jax.vmap(optimise_single_gp, in_axes=(None, 1, 0, 0))(data_x, data_y, sep_params, indices)

        return all_params, loss_vals

    def pretrain_params(self, init_data_x, init_data_y, pretrain_data_x, pretrain_data_y, key):
        params = self.create_train_state(init_data_x, init_data_y, key)
        key, _key = jrandom.split(key)
        lengthscales = jrandom.uniform(_key, minval=0.0, maxval=100, shape=(self.agent_config.PRETRAIN_RESTARTS, self.obs_dim, self.obs_dim + self.action_dim))
        # key, _key = jrandom.split(key)
        # alphas = jrandom.uniform(_key, minval=1.0, maxval=20.0, shape=(self.agent_config.PRETRAIN_RESTARTS, self.obs_dim, 1))

        def _batch_gp_training(data_x, data_y, params, lengthscales):
            kernel = gpjaxas.kernels.SeparateIndependent(
                [gpjaxas.kernels.SquaredExponential(lengthscales=lengthscales[idx], variance=self.sigma) for idx in
                 range(self.obs_dim)])
            # TODO a dodgy fix for now
            kernel_params = kernel.get_params()

            new_params = {"kernel": kernel_params, "likelihood": params["likelihood"], "mean_function": params["mean_function"]}

            return self.optimise(data_x, data_y, new_params)

        new_params, loss_vals = jax.vmap(_batch_gp_training, in_axes=(None, None, None, 0))(pretrain_data_x, pretrain_data_y, params, lengthscales)
        last_loss_vals = loss_vals[:, :, -1]
        best_idx = jnp.argmin(last_loss_vals, axis=0)
        n_indices = jnp.arange(last_loss_vals.shape[1])
        best_params = jax.tree_util.tree_map(lambda x: x[best_idx, n_indices, ...], new_params)
        kernel = gpjaxas.kernels.SeparateIndependent(
            [gpjaxas.kernels.SquaredExponential(lengthscales=best_params["kernel"]["lengthscales"][idx], variance=best_params["kernel"]["variance"][idx]) for idx in
             range(self.obs_dim)])

        kernel_params = kernel.get_params()
        params["kernel"] = kernel_params

        return params

    def get_post_mu_cov(self, XNew, params, full_cov=False):  # TODO if no data then return the prior mu and var
        mu, std = self.gp.predict_f(params, XNew, full_cov=full_cov)
        return mu, std

    def get_post_mu_cov_samples(self, XNew, params, key, full_cov=False):
        samples = self.gp.predict_f_samples(params, key, XNew, num_samples=1, full_cov=full_cov)
        return samples[0] # TODO works on the first sample as only one required for now

    def predict_on_noisy_inputs(self, m, s, train_state):
        iK, beta = self.calculate_factorizations(train_state)
        return self.predict_given_factorizations(m, s, iK, beta, train_state)

    def calculate_factorizations(self, train_state):
        X = train_state["train_data_x"]
        Y = train_state["train_data_y"]
        Kmm = self.gp.kernel(train_state["kernel"], X, X) # [..., M, M]
        batched_eye = jnp.expand_dims(jnp.eye(jnp.shape(X)[0]), axis=0).repeat(self.obs_dim, axis=0)
        L = jsp.linalg.cho_factor(Kmm + default_jitter() * batched_eye, lower=True)
        iK = jsp.linalg.cho_solve(L, batched_eye)
        Y_ = jnp.transpose(Y)[:, :, None]
        beta = jsp.linalg.cho_solve(L, Y_)[:, :, 0]
        return iK, beta

    def predict_given_factorizations(self, m, s, iK, beta, train_state):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """

        variance = jnp.tile(jnp.array(self.sigma), (self.obs_dim,))  # TODO this should be from train_state

        s = jnp.tile(s[None, None, :, :], [self.obs_dim, self.obs_dim, 1, 1])
        inp = jnp.tile(self.centralized_input(train_state["train_data_x"], m)[None, :, :], [self.obs_dim, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        # lengthscales = train_state["kernel"]  # TODO add this in to sort it out as using self.ls is bad
        iL = jax.vmap(lambda x: jnp.diag(x, k=0))(1 / self.ls)
        # iL = objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection())(1 / self.lengthscales)
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
        z = jax.vmap(jax.vmap(lambda x: jnp.diag(x, k=0)))(1.0 / jnp.square(self.ls[None, :, :]) + 1.0 / jnp.square(self.ls[:, None, :]))
        # TODO the above needs to come from train_state

        R = (s @ z) + jnp.eye(self.obs_dim + self.action_dim)

        X = inp[None, :, :, :] / jnp.square(self.ls[:, None, None, :])  # TODO this should be from train_state
        X2 = -inp[:, None, :, :] / jnp.square(self.ls[None, :, None, :])
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

    def centralized_input(self, X, m):
        return X - m
