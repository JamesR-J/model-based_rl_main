import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
from flax import struct
from flax.training import train_state
import optax
from typing import List, Tuple, Dict, Optional, NamedTuple, Any
from functools import partial
import GPJax_AScannell as gpjaxas
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
from GPJax_AScannell.gpjax.utilities.ops import sample_mvn_diag, sample_mvn
from cola.ops.operators import I_like
from flax import nnx
import tensorflow_probability.substrates.jax as tfp


class SeparateIndependent(gpjax.kernels.stationary.StationaryKernel):
    def __init__(self, lengthscale1, lengthscale2, variance1, variance2):
        self.kernel0 = gpjax.kernels.RBF(active_dims=[0, 1, 2], lengthscale=lengthscale1, variance=variance1)
        self.kernel1 = gpjax.kernels.RBF(active_dims=[0, 1, 2], lengthscale=lengthscale2, variance=variance2)
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


class MOGPGPJax(DynamicsModelBase):
    def __init__(self, env, env_params, config, agent_config, key):
        super().__init__(env, env_params, config, agent_config, key)

        kernel = SeparateIndependent(lengthscale1 = jnp.array((2.81622296,   9.64035469, 142.60660018)),
                                     lengthscale2 = jnp.array((0.92813981, 280.24169475,  14.85778016)),
                                     variance1 = jnp.array((0.78387795)),
                                     variance2 = jnp.array((0.22877621)))

        # mean = gpjax.mean_functions.Zero()
        mean = gpjax.mean_functions.Constant(jnp.array((0.07455202985890419)))
        self.prior = gpjax.gps.Prior(mean_function=mean, kernel=kernel)
        self.likelihood_builder = lambda n: gpjax.likelihoods.Gaussian(num_datapoints=n,
                                                                       obs_stddev=gpjax.parameters.PositiveReal(jnp.array(0.005988507226896687)))

    def create_train_state(self, init_data_x, init_data_y, key):
        data = self._adjust_dataset(gpjax.Dataset(init_data_x, init_data_y))
        likelihood = self.likelihood_builder(data.n)
        posterior = self.prior * likelihood
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

        assert new_x.shape == (num_points * out_dim, in_dim + 1), "Output X is the wrong shape"
        assert new_y.shape == (num_points * out_dim, 1), "Output Y is the wrong shape"

        return gpjax.Dataset(new_x, new_y)

    def pretrain_params(self, init_data_x, init_data_y, pretrain_data_x, pretrain_data_y, key):
        opt_posterior = self.optimise_gp(pretrain_data_x, pretrain_data_y, key)

        lengthscales = {}
        variances = {}
        for i in range(self.obs_dim):
            lengthscales["GP" + str(i)] = opt_posterior["train_state"]["prior"]["kernel"][f"kernel{i}"]["lengthscale"].value
            variances["GP" + str(i)] = opt_posterior["train_state"]["prior"]["kernel"][f"kernel{i}"]["variance"].value

        logging.info("-----Pretrained GP Params------")
        logging.info("---Lengthscales---")
        logging.info(lengthscales)
        logging.info("---Variances---")
        logging.info(variances)
        logging.info("---Likelihood Stddev---")
        logging.info(opt_posterior["train_state"]["likelihood"]["obs_stddev"].value)
        logging.info("---Mean Function---")
        logging.info(opt_posterior["train_state"]["prior"]["mean_function"]["constant"].value)

        return opt_posterior

    @partial(jax.jit, static_argnums=(0,))
    def optimise_gp(self, x, y, key):
        key, _key = jrandom.split(key)
        data = self._adjust_dataset(gpjax.Dataset(x, y))
        likelihood = self.likelihood_builder(data.n)

        opt_posterior, _ = gpjax.fit(model=self.prior * likelihood,
                                     objective=lambda p, d: -gpjax.objectives.conjugate_mll(p, d),
                                     train_data=data,
                                     optim=optax.adam(learning_rate=self.agent_config.GP_LR),
                                     num_iters=self.agent_config.TRAIN_GP_NUM_ITERS,
                                     safe=True,
                                     key=_key,
                                     verbose=False)

        graphdef, state = nnx.split(opt_posterior)

        return {"train_state": state}

    @partial(jax.jit, static_argnums=(0,))
    def get_post_mu_cov(self, XNew, params, train_data, full_cov=False):  # TODO if no data then return the prior mu and var
        data = self._adjust_dataset(train_data)

        likelihood = self.likelihood_builder(data.n)

        graphdef, state = nnx.split(self.prior * likelihood)
        opt_posterior = nnx.merge(graphdef, params["train_state"])

        XNew3D = self._adjust_dataset(gpjax.Dataset(XNew, jnp.zeros((XNew.shape[0], 2))))  # TODO separate this to be just X aswell

        latent_dist = opt_posterior.predict(XNew3D.X, data)
        mu = latent_dist.mean()  # TODO I think this is pedict_f, predict_y would be passing the latent dist to the posterior.likelihood
        mu = mu.reshape(-1, self.obs_dim) # TODO is this correct?

        std = latent_dist.stddev()
        std = std.reshape(-1, self.obs_dim) # TODO is this correct?

        return mu, std

    @partial(jax.jit, static_argnums=(0,))
    def get_post_mu_fullcov(self, XNew, params, train_data, full_cov=False):  # TODO if no data then return the prior mu and var
        data = self._adjust_dataset(train_data)

        likelihood = self.likelihood_builder(data.n)

        graphdef, state = nnx.split(self.prior * likelihood)
        opt_posterior = nnx.merge(graphdef, params["train_state"])

        XNew3D = self._adjust_dataset(gpjax.Dataset(XNew, jnp.zeros((XNew.shape[0], 2))))  # TODO separate this to be just X aswell

        latent_dist = opt_posterior.predict(XNew3D.X, data)
        mu = latent_dist.mean()  # TODO I think this is pedict_f, predict_y would be passing the latent dist to the posterior.likelihood
        mu = mu.reshape(-1, self.obs_dim) # TODO is this correct?

        cov = latent_dist.covariance()
        cov = jnp.expand_dims(cov, axis=0)  # cov.reshape((XNew.shape[0], -1))  # TODO a dodgy fix

        return mu, cov

    @partial(jax.jit, static_argnums=(0,))
    def get_post_mu_cov_samples(self, XNew, params, train_data, key, full_cov=False):
        data = self._adjust_dataset(train_data)

        likelihood = self.likelihood_builder(data.n)
        posterior = self.prior * likelihood

        graphdef, state = nnx.split(posterior)
        opt_posterior = nnx.merge(graphdef, params["train_state"])

        XNew3D = self._adjust_dataset(gpjax.Dataset(XNew, jnp.zeros((XNew.shape[0], 2))))  # TODO separate this to be just X aswell

        latent_dist = opt_posterior.predict(XNew3D.X, data)
        key, _key = jrandom.split(key)
        samples = latent_dist.sample(_key, (1,))
        # samples = latent_dist.sample(params["sample_key"], (1,))

        return samples
