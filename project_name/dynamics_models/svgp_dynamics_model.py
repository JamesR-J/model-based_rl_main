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


class MOSVGP(DynamicsModelBase):
    def __init__(self, env, env_params, config, agent_config, key):
        super().__init__(env, env_params, config, agent_config, key)
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

        key, _key = jrandom.split(key)
        samples = jrandom.uniform(key, shape=(self.agent_config.NUM_INDUCING_POINTS, self.obs_dim + self.action_dim),
                                  minval=0.0, maxval=1.0)
        low = jnp.concatenate([env.observation_space(env_params).low,
                               jnp.expand_dims(jnp.array(env.action_space(env_params).low), axis=0)])
        high = jnp.concatenate([env.observation_space(env_params).high,
                                jnp.expand_dims(jnp.array(env.action_space(env_params).high), axis=0)])
        # TODO this is general maybe can put somehwere else
        inducing_variable = low + (high - low) * samples

        self.gp = gpjaxas.models.SVGP(self.kernel, self.likelihood, inducing_variable, self.mean_function, num_latent_gps=self.obs_dim)

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

        transforms = self.gp.get_transforms()
        constrain_params = gpjaxas.parameters.build_constrain_params(transforms)

        learning_rate = 1e-3
        # learning_rate = 1e-2
        num_epochs = 900

        # Create optimizer
        tx = optax.adam(learning_rate)
        opt_state = tx.init(params)

        def negative_elbo(params, batch):
            return -self.gp.build_elbo(constrain_params=constrain_params, num_data=init_data_x.shape[0])(params, batch)

        @jax.jit
        def train_step(params, opt_state, batch):
            loss_val, loss_gradient = jax.value_and_grad(negative_elbo)(params, batch)
            updates, opt_state = tx.update(loss_gradient, opt_state, params)
            params = optax.apply_updates(params, updates)

            return loss_val, params, opt_state

        for epoch in range(num_epochs):
            loss, params, opt_state = train_step(params, opt_state, (pretrain_data_x, pretrain_data_y))

        print("HERE")






        # def _batch_gp_training(data_x, data_y, params, lengthscales):
        #     kernel = gpjaxas.kernels.SeparateIndependent(
        #         [gpjaxas.kernels.SquaredExponential(lengthscales=lengthscales[idx], variance=self.sigma) for idx in
        #          range(self.obs_dim)])
        #     # TODO a dodgy fix for now
        #     kernel_params = kernel.get_params()
        #
        #     params["kernel"] = kernel_params
        #
        #     return self.optimise(data_x, data_y, params)
        #
        # new_params, loss_vals = jax.vmap(_batch_gp_training, in_axes=(None, None, None, 0))(pretrain_data_x, pretrain_data_y, params, lengthscales)
        # last_loss_vals = loss_vals[:, :, -1]
        # best_idx = jnp.argmin(last_loss_vals, axis=0)
        # n_indices = jnp.arange(last_loss_vals.shape[1])
        # best_params = jax.tree_util.tree_map(lambda x: x[best_idx, n_indices, ...], new_params)
        # kernel = gpjaxas.kernels.SeparateIndependent(
        #     [gpjaxas.kernels.SquaredExponential(lengthscales=best_params["kernel"]["lengthscales"][idx], variance=best_params["kernel"]["variance"][idx]) for idx in
        #      range(self.obs_dim)])
        # # TODO again a dodgy fix for now
        #
        # kernel_params = kernel.get_params()
        # params["kernel"] = kernel_params

        return params

    def get_post_mu_cov(self, XNew, params, full_cov=False):  # TODO if no data then return the prior mu and var
        mu, std = self.gp.predict_f(params, XNew, full_cov=full_cov)
        return mu, std

    def get_post_mu_cov_samples(self, XNew, params, key, full_cov=False):
        samples = self.gp.predict_f_samples(params, key, XNew, num_samples=1, full_cov=full_cov)
        return samples[0] # TODO works on the first sample as only one required for now

    def predict_on_noisy_inputs(self, m, s, train_state):
        iK, beta = self.calculate_factorizations()
        return self.predict_given_factorizations(m, s, iK, beta)

    def calculate_factorizations(self):
        K = self.K(self.X, self.X)
        batched_eye = jnp.expand_dims(jnp.eye(jnp.shape(self.X)[0]), axis=0).repeat(
            self.num_outputs, axis=0
        )
        L = jsp.linalg.cho_factor(
            K + self.noise[:, None, None] * batched_eye, lower=True
        )
        iK = jsp.linalg.cho_solve(L, batched_eye)
        Y_ = jnp.transpose(self.Y)[:, :, None]
        beta = jsp.linalg.cho_solve(L, Y_)[:, :, 0]
        return iK, beta

    def predict_given_factorizations(self, m, s, iK, beta):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """

        s = jnp.tile(s[None, None, :, :], [self.num_outputs, self.num_outputs, 1, 1])
        inp = jnp.tile(self.centralized_input(m)[None, :, :], [self.num_outputs, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection())(
            1 / self.lengthscales
        )
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + jnp.eye(self.num_dims)

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = jnp.transpose(
            jnp.linalg.solve(B, jnp.transpose(iN, axes=(0, 2, 1))),
            axes=(0, 2, 1),
        )

        lb = jnp.exp(-0.5 * jnp.sum(iN * t, -1)) * beta
        tiL = t @ iL
        c = self.variance / jnp.sqrt(jnp.linalg.det(B))

        M = (jnp.sum(lb, -1) * c)[:, None]
        V = (jnp.transpose(tiL, axes=(0, 2, 1)) @ lb[:, :, None])[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        z = objax.Vectorize(
            objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection()),
            objax.VarCollection(),
        )(
            1.0 / jnp.square(self.lengthscales[None, :, :])
            + 1.0 / jnp.square(self.lengthscales[:, None, :])
        )

        R = (s @ z) + jnp.eye(self.num_dims)

        X = inp[None, :, :, :] / jnp.square(self.lengthscales[:, None, None, :])
        X2 = -inp[:, None, :, :] / jnp.square(self.lengthscales[None, :, None, :])
        Q = 0.5 * jnp.linalg.solve(R, s)
        maha = (X - X2) @ Q @ jnp.transpose(X - X2, axes=(0, 1, 3, 2))

        k = jnp.log(self.variance)[:, None] - 0.5 * jnp.sum(jnp.square(iN), -1)
        L = jnp.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = (
            jnp.tile(beta[:, None, None, :], [1, self.num_outputs, 1, 1])
            @ L
            @ jnp.tile(beta[None, :, :, None], [self.num_outputs, 1, 1, 1])
        )[:, :, 0, 0]

        diagL = jnp.transpose(
            objax.Vectorize(
                objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection()),
                objax.VarCollection(),
            )(jnp.transpose(L))
        )
        S = S - jnp.diag(jnp.sum(jnp.multiply(iK, diagL), [1, 2]))
        S = S / jnp.sqrt(jnp.linalg.det(R))
        S = S + jnp.diag(self.variance)
        S = S - M @ jnp.transpose(M)

        return jnp.transpose(M), S, jnp.transpose(V)
