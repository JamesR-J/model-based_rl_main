import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
# import gpjax
from flax import struct
from flax.training import train_state
import optax
from typing import List, Tuple, Dict, Optional, NamedTuple, Any
from functools import partial
import GPJax_AScannell as gpjaxas
import optax
# import gpjax
from project_name.dynamics_models import DynamicsModelBase
from gpjax.typing import (
    Array,
    ScalarFloat,
)
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.kernels.stationary.utils import squared_distance
from jaxtyping import Float, install_import_hook
from gpjax.kernels.computations import DenseKernelComputation
from gpjax.parameters import Static
from GPJax_AScannell.gpjax.utilities.ops import sample_mvn_diag, sample_mvn
from cola.ops.operators import I_like

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax


# class SeparateIndependent(gpjax.kernels.AbstractKernel):
#     """Separate independent kernels for each output dimension"""
#
#     name: str = "SeparateIndependent"
#
#     def __init__(self,
#                  kernel0: gpjax.kernels.stationary.StationaryKernel = gpjax.kernels.RBF(active_dims=[0, 1, 2], lengthscale=jnp.array([10.0, 8.0, 10.0]), variance=25.0),
#                  kernel1: gpjax.kernels.stationary.StationaryKernel = gpjax.kernels.RBF(active_dims=[0, 1, 2], lengthscale=jnp.array([10.0, 8.0, 10.0]), variance=25.0)
#                  ):
#         self.kernel0 = kernel0
#         self.kernel1 = kernel1
#         super().__init__(compute_engine=DenseKernelComputation())
#
#     def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
#         Kxxs = jax.tree_map(lambda kern: kern(x, y), [self.kernel0, self.kernel1])
#         Kxxs = jnp.stack(Kxxs, axis=-1)
#         return Kxxs


class SeparateIndependent(gpjax.kernels.AbstractKernel):
    def __init__(self,
                 kernel0: gpjax.kernels.stationary.StationaryKernel = gpjax.kernels.RBF(active_dims=[0, 1, 2], lengthscale=jnp.array([2.27, 7.73, 138.94]), variance=0.01),
                 kernel1: gpjax.kernels.stationary.StationaryKernel = gpjax.kernels.RBF(active_dims=[0, 1, 2], lengthscale=jnp.array([0.84, 288.15, 11.05]), variance=0.01)
                 ):
        self.kernel0 = kernel0
        self.kernel1 = kernel1
        super().__init__(compute_engine=DenseKernelComputation())

    def __call__(self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        # standard RBF-SE kernel is x and x' are on the same output, otherwise returns 0

        z = jnp.array(X[2], dtype=int)
        zp = jnp.array(Xp[2], dtype=int)

        # achieve the correct value via 'switches' that are either 1 or 0
        k0_switch = ((z + 1) % 2) * ((zp + 1) % 2)
        k1_switch = z * zp

        return k0_switch * self.kernel0(X, Xp) + k1_switch * self.kernel1(X, Xp)



class MOGPGPJax(DynamicsModelBase):
    def __init__(self, env, env_params, config, agent_config, key):
        super().__init__(env, env_params, config, agent_config, key)
        num_latent_gps = self.obs_dim

        self.ls = jnp.array([[2.27, 7.73, 138.94], [0.84, 288.15, 11.05]],
                            dtype=jnp.float64)
        alpha = jnp.array([0.26, 2.32, 11.59, 3.01], dtype=jnp.float64)  # TODO what is alpha? is this periodic?
        self.sigma = 0.01

        mean = gpjax.mean_functions.Zero()
        # kernel = gpx.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0)
        kernel = SeparateIndependent()
        self.prior = gpjax.gps.Prior(mean_function=mean, kernel=kernel)

        key, _key = jrandom.split(key)
        samples = jrandom.uniform(key, shape=(self.agent_config.NUM_INDUCING_POINTS, self.obs_dim + self.action_dim), minval=0.0, maxval=1.0)
        low = jnp.concatenate([env.observation_space(env_params).low,
                               jnp.expand_dims(jnp.array(env.action_space(env_params).low), axis=0)])
        high = jnp.concatenate([env.observation_space(env_params).high,
                                jnp.expand_dims(jnp.array(env.action_space(env_params).high), axis=0)])
        # TODO this is general maybe can put somehwere else
        self.z = low + (high - low) * samples

    def create_train_state(self, init_data_x, init_data_y, key):
        params = {}
        params["data"] = self._adjust_dataset(init_data_x, init_data_y)
        params["inducing_points"] = self.z
        return params

    @staticmethod
    def _adjust_dataset(x, y):  # TODO generalise this to more dimensions
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

    def optimise_gp(self, x, y, key):
        key, _key = jrandom.split(key)
        data = self._adjust_dataset(x, y)
        likelihood = gpjax.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=Static(jnp.array(1e-6)))
        posterior = self.prior * likelihood
        q = gpjax.variational_families.VariationalGaussian(posterior, self.z)

        # TODO do we add a scheduling lr?
        opt_posterior, _ = gpjax.fit(model=q,
                                     objective=lambda p, d: -gpjax.objectives.elbo(p, d),
                                     train_data=data,
                                     optim=optax.adam(learning_rate=self.agent_config.GP_LR),
                                     num_iters=1000,
                                     safe=True,
                                     key=_key,
                                     verbose=False)

        return opt_posterior

    def predict_on_noisy_inputs(self, m, s, params):   # TODO Idk if even nee this
        data = params["data"]  # self._adjust_dataset(params["train_data_x"], params["train_data_y"])

        # turns dataset of data_points, num_features into data_points * num_outputs, num_features + (num_outputs - 1)
        # TODO generalise the above thing to make it actually work with n number of outputs

        likelihood = gpjax.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=Static(jnp.array(1e-6)))
        posterior = self.prior * likelihood
        q = gpjax.variational_families.VariationalGaussian(posterior, params["inducing_points"])

        # XNew3D = self._adjust_dataset(XNew, jnp.zeros((XNew.shape[0], 2)))  # TODO separate this to be just X aswell

        data_x = jnp.ones((160, 3))
        data = self._adjust_dataset(data_x, jnp.ones((160, 2)))  # TODO how do we do this if using the gpjax dataset all the time?
        new_data = self._adjust_dataset(m, jnp.zeros((m.shape[0], 2)))

        def calculate_factorisations(posterior, data):
            K = posterior.prior.kernel.gram(data.X).A
            eye = jnp.eye(jnp.shape(data.X)[0])
            L = jsp.linalg.cho_factor(K + eye * posterior.jitter, lower=True)
            iK = jsp.linalg.cho_solve(L, eye)
            beta = jsp.linalg.cho_solve(L, data.y)[:, 0]
            return iK, beta

        def predict_given_factorizations(m, s, inp, iK, beta, data):
            """
            Approximate GP regression at noisy inputs via moment matching
            IN: mean (m) (row vector) and (s) variance of the state
            OUT: mean (M) (row vector), variance (S) of the action
                 and inv(s)*input-ouputcovariance
            """
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
            z = objax.Vectorize(objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection()), objax.VarCollection())(
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

        iK, beta = calculate_factorisations(q, data)
        yeyo = predict_given_factorizations(new_data.X, s, data_x - m, iK, beta, data)

    def get_post_mu_cov(self, XNew, params, full_cov=False):  # TODO if no data then return the prior mu and var
        data = self._adjust_dataset(params["train_data_x"], params["train_data_y"])

        # turns dataset of data_points, num_features into data_points * num_outputs, num_features + (num_outputs - 1)
        # TODO generalise the above thing to make it actually work with n number of outputs

        likelihood = gpjax.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=Static(jnp.array(1e-6)))
        posterior = self.prior * likelihood
        q = gpjax.variational_families.VariationalGaussian(posterior, self.z)

        XNew3D = self._adjust_dataset(XNew, jnp.zeros((XNew.shape[0], 2)))  # TODO separate this to be just X aswell

        latent_dist = q.predict(XNew3D.X, data)
        mu = latent_dist.mean()  # TODO I think this is pedict_f, predict_y would be passing the latent dist to the posterior.likelihood
        std = latent_dist.stddev()

        mu = mu.reshape((XNew.shape[0], -1))  # TODO a dodgy fix, is this correct?
        std = std.reshape((XNew.shape[0], -1))

        return mu, std

    def get_post_mu_full_cov(self, XNew, params, full_cov=False):  # TODO if no data then return the prior mu and var
        data = gpjax.Dataset(X=params["train_data_x"], y=params["train_data_y"])
        data = self.adjust_dataset(params["train_data_x"], params["train_data_y"])

        # turns dataset of data_points, num_features into data_points * num_outputs, num_features + (num_outputs - 1)
        # TODO generalise the above thing to make it actually work with n number of outputs

        likelihood = gpjax.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=Static(jnp.array(1e-6)))
        posterior = self.prior_gpjax * likelihood
        q = gpjax.variational_families.VariationalGaussian(posterior, self.z)

        XNew3D = self.adjust_dataset(XNew, jnp.zeros((XNew.shape[0], 2)))  # TODO separate this to be just X aswell

        latent_dist = posterior.predict(XNew3D.X, data)
        mu = latent_dist.mean()  # TODO I think this is pedict_f, predict_y would be passing the latent dist to the posterior.likelihood
        cov = latent_dist.covariance()

        mu = mu.reshape((XNew.shape[0], -1))  # TODO a dodgy fix, is this correct?
        cov = jnp.expand_dims(cov, axis=0)  # cov.reshape((XNew.shape[0], -1))  # TODO a dodgy fix

        return mu, cov

    def get_post_mu_cov_samples(self, XNew, params, key, full_cov=False):
        data = gpjax.Dataset(X=params["train_data_x"], y=params["train_data_y"])
        data = self.adjust_dataset(params["train_data_x"], params["train_data_y"])

        # turns dataset of data_points, num_features into data_points * num_outputs, num_features + (num_outputs - 1)
        # TODO generalise the above thing to make it actually work with n number of outputs

        likelihood = gpjax.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=Static(jnp.array(1e-6)))
        posterior = self.prior_gpjax * likelihood

        XNew3D = self.adjust_dataset(XNew, jnp.zeros((XNew.shape[0], 2)))  # TODO separate this to be just X aswell

        latent_dist = posterior.predict(XNew3D.X, data)
        samples = latent_dist.sample(key, (1,))

        return samples
