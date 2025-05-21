import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
from flax import struct
from flax.training import train_state
import optax
from typing import List, Tuple, Dict, Optional, NamedTuple, Any
from functools import partial
import optax
from project_name.dynamics_models import DynamicsModelBase
from jaxtyping import Float, install_import_hook

with install_import_hook("gpjax", "beartype.beartype"):
    import logging
    logging.getLogger('gpjax').setLevel(logging.WARNING)
    import gpjax

from gpjax.typing import (
    Array,
    ScalarFloat,
)
from cola.ops.operators import I_like
from flax import nnx
import tensorflow_probability.substrates.jax as tfp


class SeparateIndependent(gpjax.kernels.stationary.StationaryKernel):
    def __init__(self, lengthscale1, lengthscale2, variance1, variance2):
        self.kernel0 = gpjax.kernels.Matern52(active_dims=[0, 1, 2], lengthscale=lengthscale1, variance=variance1)
        self.kernel1 = gpjax.kernels.Matern52(active_dims=[0, 1, 2], lengthscale=lengthscale2, variance=variance2)
        super().__init__(n_dims=3, compute_engine=gpjax.kernels.computations.DenseKernelComputation())

    def __call__(self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        # standard RBF-SE kernel is x and x' are on the same output, otherwise returns 0

        z = jnp.array(X[-1], dtype=int)
        zp = jnp.array(Xp[-1], dtype=int)

        # achieve the correct value via 'switches' that are either 1 or 0
        k0_switch = ((z + 1) % 2) * ((zp + 1) % 2)
        k1_switch = z * zp

        return k0_switch * self.kernel0(X, Xp) + k1_switch * self.kernel1(X, Xp)

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)


class MOSVGP(DynamicsModelBase):
    def __init__(self, env, config, agent_config, key):
        super().__init__(env, config, agent_config, key)

        key, _key = jrandom.split(key)
        samples = jrandom.uniform(key, shape=(self.agent_config.NUM_INDUCING_POINTS, self.obs_dim + self.action_dim),
                                  minval=0.0, maxval=1.0)
        low = jnp.concatenate([env.observation_space().low,
                               jnp.expand_dims(jnp.array(env.action_space().low), axis=0)])
        high = jnp.concatenate([env.observation_space().high,
                                jnp.expand_dims(jnp.array(env.action_space().high), axis=0)])
        # TODO this is general maybe can put somewhere else
        self.og_z = low + (high - low) * samples
        self.z = self._adjust_dataset(gpjax.Dataset(self.og_z, jnp.zeros((self.og_z.shape[0], self.obs_dim))))

        # kernel = SeparateIndependent(lengthscale1 = jnp.array((2.81622296,   9.64035469, 142.60660018)),
        #                              lengthscale2 = jnp.array((0.92813981, 280.24169475,  14.85778016)),
        #                              variance1 = jnp.array((0.78387795)),
        #                              variance2 = jnp.array((0.22877621)))

        kernel = SeparateIndependent(lengthscale1=jnp.ones((3,)),
                                     lengthscale2=jnp.ones((3,)),
                                     variance1=1.0,
                                     variance2=1.0)

        # mean = gpjax.mean_functions.Zero()
        mean = gpjax.mean_functions.Constant(jnp.array((0.07455202985890419)))
        prior = gpjax.gps.Prior(mean_function=mean, kernel=kernel)
        self.variational_posterior_builder = lambda n: gpjax.variational_families.VariationalGaussian(posterior=prior * gpjax.likelihoods.Gaussian(num_datapoints=n,
                                                                       obs_stddev=jnp.sqrt(0.01)), inducing_inputs=self.z.X)


    def create_train_state(self, init_data, key):
        data = self._adjust_dataset(init_data)
        posterior = self.variational_posterior_builder(data.n)
        graphdef, state = nnx.split(posterior)

        return {"train_state": state}

    @staticmethod
    def _adjust_dataset(dataset):
        num_points = dataset.X.shape[0]
        in_dim = dataset.X.shape[1]
        out_dim = dataset.y.shape[1]
        # print(f"Num Points: {num_points}; Num Inputs: {in_dim}; Num Outputs: {out_dim}")

        label = jnp.tile(jnp.array(jnp.linspace(0, out_dim - 1, out_dim)), num_points)

        new_x = jnp.hstack((jnp.repeat(dataset.X, repeats=out_dim, axis=0), jnp.expand_dims(label, axis=-1)))

        new_y = dataset.y.reshape(-1, 1)

        # assert new_x.shape == (num_points * out_dim, in_dim + 1), "Output X is the wrong shape"
        # assert new_y.shape == (num_points * out_dim, 1), "Output Y is the wrong shape"
        # TODO add in assertion

        return gpjax.Dataset(new_x, new_y)

    def pretrain_params(self, init_data, pretrain_data, key):
        key, _key = jrandom.split(key)
        params = self.create_train_state(init_data, _key)
        key, _key = jrandom.split(key)
        opt_posterior = self.optimise_gp(pretrain_data, params, _key)

        # TODO should we make this a randomised beginning and couple of restarts or is this done natively in gpjax?

        lengthscales = {}
        variances = {}
        for i in range(self.obs_dim):
            lengthscales["GP" + str(i)] = opt_posterior["train_state"]["posterior"]["prior"]["kernel"][f"kernel{i}"]["lengthscale"].value
            variances["GP" + str(i)] = opt_posterior["train_state"]["posterior"]["prior"]["kernel"][f"kernel{i}"]["variance"].value

        logging.info("-----Pretrained GP Params------")
        logging.info("---Lengthscales---")
        logging.info(lengthscales)
        logging.info("---Variances---")
        logging.info(variances)
        logging.info("---Likelihood Stddev---")
        logging.info(opt_posterior["train_state"]["posterior"]["likelihood"]["obs_stddev"].value)
        logging.info("---Mean Function---")
        logging.info(opt_posterior["train_state"]["posterior"]["prior"]["mean_function"]["constant"].value)

        return opt_posterior

    # TODO does this also need to be different to respect the moment matching nature?
    @partial(jax.jit, static_argnums=(0,))
    def optimise_gp(self, opt_data, params, key):
        key, _key = jrandom.split(key)
        data = self._adjust_dataset(opt_data)
        q = self.variational_posterior_builder(data.n)

        graphdef, state = nnx.split(q)
        q = nnx.merge(graphdef, params["train_state"])

        schedule = optax.warmup_cosine_decay_schedule(init_value=0.0,
                                                      peak_value=0.02,
                                                      warmup_steps=75,
                                                      decay_steps=2000,
                                                      end_value=0.001)
        # TODO idk if we need the above

        opt_posterior, _ = gpjax.fit(model=q,
                                     objective=lambda p, d: -gpjax.objectives.elbo(p, d),
                                     train_data=data,
                                     optim=optax.adam(learning_rate=self.agent_config.GP_LR),
                                     # optim=optax.adam(learning_rate=schedule),
                                     num_iters=self.agent_config.TRAIN_GP_NUM_ITERS,
                                     batch_size=128,
                                     safe=True,
                                     key=_key,
                                     verbose=False)

        graphdef, state = nnx.split(opt_posterior)

        return {"train_state": state}

    @partial(jax.jit, static_argnums=(0,))
    def get_post_mu_cov(self, XNew, params, train_data, full_cov=False):  # TODO if no data then return the prior mu and var
        data = self._adjust_dataset(train_data)

        q = self.variational_posterior_builder(data.n)

        graphdef, state = nnx.split(q)
        opt_posterior = nnx.merge(graphdef, params["train_state"])

        XNew3D = self._adjust_dataset(gpjax.Dataset(XNew, jnp.zeros((XNew.shape[0], 2))))  # TODO separate this to be just X aswell

        latent_dist = opt_posterior.predict(XNew3D.X)
        mu = latent_dist.mean  # TODO I think this is pedict_f, predict_y would be passing the latent dist to the posterior.likelihood
        mu = mu.reshape(-1, self.obs_dim)

        std = jnp.sqrt(latent_dist.variance)
        std = std.reshape(-1, self.obs_dim)

        return mu, std

    @partial(jax.jit, static_argnums=(0,))
    def get_post_mu_fullcov(self, XNew, params, train_data, full_cov=False):  # TODO if no data then return the prior mu and var
        data = self._adjust_dataset(train_data)

        q = self.variational_posterior_builder(data.n)

        graphdef, state = nnx.split(q)
        opt_posterior = nnx.merge(graphdef, params["train_state"])

        XNew3D = self._adjust_dataset(gpjax.Dataset(XNew, jnp.zeros((XNew.shape[0], 2))))  # TODO separate this to be just X aswell

        latent_dist = opt_posterior.predict(XNew3D.X)
        mu = latent_dist.mean  # TODO I think this is pedict_f, predict_y would be passing the latent dist to the posterior.likelihood
        mu = mu.reshape(-1, self.obs_dim) # TODO is this correct?

        cov = latent_dist.covariance()
        cov = jnp.expand_dims(cov, axis=0)  # cov.reshape((XNew.shape[0], -1))  # TODO a dodgy fix

        return mu, cov

    @partial(jax.jit, static_argnums=(0,))
    def get_post_mu_cov_samples(self, XNew, params, train_data, key, full_cov=False):
        data = self._adjust_dataset(train_data)

        q = self.variational_posterior_builder(data.n)

        graphdef, state = nnx.split(q)
        opt_posterior = nnx.merge(graphdef, params["train_state"])

        XNew3D = self._adjust_dataset(gpjax.Dataset(XNew, jnp.zeros((XNew.shape[0], 2))))  # TODO separate this to be just X aswell

        latent_dist = opt_posterior.predict(XNew3D.X)
        key, _key = jrandom.split(key)
        samples = latent_dist.sample(_key, (1,))
        # samples = latent_dist.sample(params["sample_key"], (1,))

        return samples

    @partial(jax.jit, static_argnums=(0,))
    def predict_on_noisy_inputs(self, m, s, params, train_data):   # TODO Idk if even nee this
        adj_data = self._adjust_dataset(train_data)

        q = self.variational_posterior_builder(adj_data.n)

        graphdef, state = nnx.split(q)
        opt_posterior = nnx.merge(graphdef, params["train_state"])

        K = opt_posterior.posterior.prior.kernel.gram(adj_data.X).A

        def reformat_K(matrix, k=2):
            matrix1 = matrix[::k, ::k]
            matrix2 = matrix[1::k, 1::k]
            return jnp.concatenate((jnp.expand_dims(matrix1, axis=0), jnp.expand_dims(matrix2, axis=0)), axis=0)

        K = reformat_K(K)
        batched_eye = jnp.expand_dims(jnp.eye(jnp.shape(train_data.X)[0]), axis=0).repeat(self.output_dim, axis=0)
        obs_noise = jnp.repeat(opt_posterior.posterior.likelihood.obs_stddev.value ** 2, self.output_dim, axis=0)
        L = jsp.linalg.cho_factor(K + obs_noise[:, None, None] * batched_eye, lower=True)
        iK = jsp.linalg.cho_solve(L, batched_eye)
        Y_ = jnp.transpose(train_data.y)[:, :, None]
        beta = jsp.linalg.cho_solve(L, Y_)[:, :, 0]

        m, s, c = self._predict_given_factorisations(m, s,  iK, beta, train_data, params)
        return m, s, c

    @partial(jax.jit, static_argnums=(0,))
    def _predict_given_factorisations(self, m, s, iK, beta, unadj_data, params):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """
        s = jnp.tile(s[None, None, :, :], [self.output_dim, self.output_dim, 1, 1])  # 2, 2, 3, 3
        inp = jnp.tile(self._centralised_input(unadj_data.X, m)[None, :, :], [self.output_dim, 1, 1])

        lengthscales = jnp.concatenate((jnp.expand_dims(params["train_state"]["posterior"]["prior"]["kernel"]["kernel0"]["lengthscale"].value, axis=0),
                                        jnp.expand_dims(params["train_state"]["posterior"]["prior"]["kernel"]["kernel0"]["lengthscale"].value, axis=0)))
        variance = jnp.concatenate((jnp.expand_dims(params["train_state"]["posterior"]["prior"]["kernel"]["kernel0"]["variance"].value, axis=0),
                                    jnp.expand_dims(params["train_state"]["posterior"]["prior"]["kernel"]["kernel0"]["variance"].value, axis=0)))
        # TODO generalise the above

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = jax.vmap(lambda x: jnp.diag(x, k=0))(1 / lengthscales)
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + jnp.eye(self.input_dim)

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = jnp.transpose(jnp.linalg.solve(B, jnp.transpose(iN, axes=(0, 2, 1))), axes=(0, 2, 1))

        lb = jnp.exp(-0.5 * jnp.sum(iN * t, -1)) * beta
        tiL = t @ iL
        c = variance / jnp.sqrt(jnp.linalg.det(B))

        M = (jnp.sum(lb, -1) * c)[:, None]
        V = (jnp.transpose(tiL, axes=(0, 2, 1)) @ lb[:, :, None])[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        z = jax.vmap(jax.vmap(lambda x: jnp.diag(x, k=0)))(
            1.0 / jnp.square(lengthscales[None, :, :]) + 1.0 / jnp.square(lengthscales[:, None, :]))

        R = (s @ z) + jnp.eye(self.obs_dim + self.action_dim)

        X = inp[None, :, :, :] / jnp.square(lengthscales[:, None, None, :])
        X2 = -inp[:, None, :, :] / jnp.square(lengthscales[None, :, None, :])
        Q = 0.5 * jnp.linalg.solve(R, s)
        maha = (X - X2) @ Q @ jnp.transpose(X - X2, axes=(0, 1, 3, 2))

        k = jnp.log(variance)[:, None] - 0.5 * jnp.sum(jnp.square(iN), -1)
        L = jnp.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = (jnp.tile(beta[:, None, None, :], [1, self.obs_dim, 1, 1]) @ L
             @ jnp.tile(beta[None, :, :, None], [self.obs_dim, 1, 1, 1]))[:, :, 0, 0]

        diagL = jnp.transpose(jax.vmap(jax.vmap(lambda x: jnp.diag(x, k=0)))(jnp.transpose(L)))
        S = S - jnp.diag(jnp.sum(jnp.multiply(iK, diagL), [1, 2]))
        S = S / jnp.sqrt(jnp.linalg.det(R))
        S = S + jnp.diag(variance)
        S = S - M @ jnp.transpose(M)

        S = jnp.clip(S, -1, 1)

        return jnp.transpose(M), S, jnp.transpose(V)

    @partial(jax.jit, static_argnums=(0,))
    def _centralised_input(self, X, m):
        return X - m

