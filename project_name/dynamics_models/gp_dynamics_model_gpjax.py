import jax
import jax.numpy as jnp
import jax.random as jrandom
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

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


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


class SeparateIndependent(gpx.kernels.AbstractKernel):
    def __init__(self,
                 kernel0: gpx.kernels.stationary.StationaryKernel = gpx.kernels.RBF(active_dims=[0, 1, 2], lengthscale=jnp.array([2.27, 7.73, 138.94]), variance=0.01),
                 kernel1: gpx.kernels.stationary.StationaryKernel = gpx.kernels.RBF(active_dims=[0, 1, 2], lengthscale=jnp.array([0.84, 288.15, 11.05]), variance=0.01)
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

        mean = gpx.mean_functions.Zero()
        # kernel = gpx.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0)
        kernel = SeparateIndependent()
        self.prior_gpjax = gpx.gps.Prior(mean_function=mean, kernel=kernel)

        self.kernel = gpjaxas.kernels.SeparateIndependent([gpjaxas.kernels.SquaredExponential(lengthscales=self.ls[idx], variance=self.sigma) for idx in range(self.obs_dim)])
        self.likelihood = gpjaxas.likelihoods.Gaussian(variance=3.0)
        self.mean_function = gpjaxas.mean_functions.Zero(output_dim=self.action_dim)
        self.gp = gpjaxas.models.GPR(self.kernel, self.likelihood, self.mean_function, num_latent_gps=num_latent_gps)

    def create_train_state(self, init_data_x, init_data_y, key):
        params = self.gp.get_params()
        params["train_data_x"] = init_data_x
        params["train_data_y"] = init_data_y
        return params

    # def pretrain_params(self, init_data_x, init_data_y, key):
    #     # train on the data and update the ls and sigma etc for the rest
    #     params = self.gp.get_params()
    #     transforms = self.gp.get_transforms()  # TODO do we need this for a gp idk?
    #     constrain_params = gpjaxas.parameters.build_constrain_params(transforms)
    #
    #     learning_rate = 1e-3
    #     # learning_rate = 1e-2
    #     num_epochs = 900
    #
    #     # Create optimizer
    #     tx = optax.adam(learning_rate)
    #     opt_state = tx.init(
    #
    #
    #     # recreate a train_state if fitting params
    #     return self.create_train_state(init_data_x, init_data_y, key)

    def adjust_dataset(self, x, y):
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
            return gpx.Dataset(label_position(pos), stack_velocity(vel))

        # takes in dimension (number of data points, num features)

        return dataset_3d(jnp.swapaxes(x, 0, 1), jnp.swapaxes(y, 0, 1))

    def get_post_mu_cov(self, XNew, params, full_cov=False):  # TODO if no data then return the prior mu and var
        data = gpx.Dataset(X=params["train_data_x"], y=params["train_data_y"])
        data = self.adjust_dataset(params["train_data_x"], params["train_data_y"])

        # turns dataset of data_points, num_features into data_points * num_outputs, num_features + (num_outputs - 1)
        # TODO generalise the above thing to make it actually work with n number of outputs

        likelihood = gpx.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=Static(jnp.array(1e-6)))
        posterior = self.prior_gpjax * likelihood

        XNew3D = self.adjust_dataset(XNew, jnp.zeros((XNew.shape[0], 2)))  # TODO separate this to be just X aswell

        latent_dist = posterior.predict(XNew3D.X, data)
        mu = latent_dist.mean()  # TODO I think this is pedict_f, predict_y would be passing the latent dist to the posterior.likelihood
        std = latent_dist.stddev()

        mu = mu.reshape((XNew.shape[0], -1))  # TODO a dodgy fix, is this correct?
        std = std.reshape((XNew.shape[0], -1))

        return mu, std

    def get_post_mu_full_cov(self, XNew, params, full_cov=False):  # TODO if no data then return the prior mu and var
        data = gpx.Dataset(X=params["train_data_x"], y=params["train_data_y"])
        data = self.adjust_dataset(params["train_data_x"], params["train_data_y"])

        # turns dataset of data_points, num_features into data_points * num_outputs, num_features + (num_outputs - 1)
        # TODO generalise the above thing to make it actually work with n number of outputs

        likelihood = gpx.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=Static(jnp.array(1e-6)))
        posterior = self.prior_gpjax * likelihood

        XNew3D = self.adjust_dataset(XNew, jnp.zeros((XNew.shape[0], 2)))  # TODO separate this to be just X aswell

        latent_dist = posterior.predict(XNew3D.X, data)
        mu = latent_dist.mean()  # TODO I think this is pedict_f, predict_y would be passing the latent dist to the posterior.likelihood
        cov = latent_dist.covariance()

        mu = mu.reshape((XNew.shape[0], -1))  # TODO a dodgy fix, is this correct?
        cov = jnp.expand_dims(cov, axis=0)  # cov.reshape((XNew.shape[0], -1))  # TODO a dodgy fix

        return mu, cov

    def get_post_mu_cov_samples(self, XNew, params, key, full_cov=False):
        data = gpx.Dataset(X=params["train_data_x"], y=params["train_data_y"])
        data = self.adjust_dataset(params["train_data_x"], params["train_data_y"])

        # turns dataset of data_points, num_features into data_points * num_outputs, num_features + (num_outputs - 1)
        # TODO generalise the above thing to make it actually work with n number of outputs

        likelihood = gpx.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=Static(jnp.array(1e-6)))
        posterior = self.prior_gpjax * likelihood

        XNew3D = self.adjust_dataset(XNew, jnp.zeros((XNew.shape[0], 2)))  # TODO separate this to be just X aswell

        latent_dist = posterior.predict(XNew3D.X, data)
        samples = latent_dist.sample(key, (1,))

        return samples
