import gpjax
import optax

import jax.numpy as jnp
import jax.random as jrandom
from jax import config
from typing import (
    List,
    Tuple,
)
import jax

# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxopt import ScipyBoundedMinimize
from jaxtyping import (
    Float,
    Int,
    install_import_hook,
)
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import optax as ox
import tensorflow_probability.substrates.jax as tfp
import sys
from gpjax.parameters import Static
from gpjax.typing import (
    Array,
    FunctionalSample,
    ScalarFloat,
)
import neatplot
from typing import NamedTuple
import numpy as np
import time

config.update("jax_enable_x64", True)

key = jrandom.PRNGKey(42)

# Global variables
DEFAULT_F_IS_DIFF = True
LONG_PATH = True

cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


class ExPath(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray


class NStep(Algorithm):
    """
    An algorithm that takes n steps through a state space (and touches n+1 states).
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "NStep")
        self.params.n = getattr(params, "n", 10)
        self.params.f_is_diff = getattr(params, "f_is_diff", DEFAULT_F_IS_DIFF)
        self.params.init_x = getattr(params, "init_x", [0.0, 0.0])
        self.params.project_to_domain = getattr(params, "project_to_domain", True)
        self.params.domain = getattr(params, "domain", [[0.0, 10.0], [0.0, 10.0]])

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        len_path = len(self.exe_path.x)
        if len_path == 0:
            next_x = self.params.init_x
        elif len_path >= self.params.n + 1:
            next_x = None
        else:
            if self.params.f_is_diff:
                zip_path_end = zip(self.exe_path.x[-1], self.exe_path.y[-1])
                next_x = [xi + yi for xi, yi in zip_path_end]
            else:
                next_x = self.exe_path.y[-1]

            if self.params.project_to_domain:
                # Optionally, project to domain
                next_x = project_to_domain(next_x, self.params.domain)

        return next_x

    def get_output(self):
        """Return output based on self.exe_path."""
        return self.exe_path


def step_northwest(x_list, step_size=0.5, f_is_diff=DEFAULT_F_IS_DIFF):
    """Return x_list with a small positive value added to each element."""
    if f_is_diff:
        diffs_list = [step_size for x in x_list]
        return diffs_list
    else:
        x_list_new = [x + step_size for x in x_list]
        return x_list_new


def plot_path_2d(path, ax=None, true_path=False):
    """Plot a path through an assumed two-dimensional state space."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if true_path:
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=15, alpha=0.3)
    else:
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.5)


def plot_path_2d_jax(path, ax=None, true_path=False):
    """Plot a path through an assumed two-dimensional state space."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    x_plot = [xi[0] for xi in path]
    y_plot = [xi[1] for xi in path]

    if true_path:
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=15)
    else:
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.3)


# -------------
# Start Script
# -------------
# Set black-box function
f = step_northwest

# Set domain
domain = [[0, 23], [0, 23]] if LONG_PATH else [[0, 10], [0, 10]]

# Set algorithm
algo_class = NStep
n_steps = 40 if LONG_PATH else 15
algo_params = {"n": n_steps, "init_x": [0.5, 0.5], "domain": domain}
algo = algo_class(algo_params)

# Set model
gp_params = {"ls": 8.0, "alpha": 5.0, "sigma": 1e-2, "n_dimx": 2}
multi_gp_params = {"n_dimy": 2, "gp_params": gp_params}
# gp_model_class = MultiGpfsGp

# # Set acqfunction  # TODO reinstate all this stuff
# acqfn_params = {"n_path": 30}
# acqfn_class = MultiBaxAcqFunction
# n_rand_acqopt = 1000
#
# Compute true path
true_algo = algo_class(algo_params)
true_path, _ = algo.run_algorithm_on_f(f)

def create_gp_models(output_dims):  # TODO can we vmap this ever?
    """Create multiple single-output GP models."""
    mean = gpjax.mean_functions.Zero()
    kernel = gpjax.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0)  # TODO make this more general
    prior = gpjax.gps.Prior(mean_function=mean, kernel=kernel)
    return [prior for _ in range(output_dims)]

def return_optimised_posterior(data: gpjax.Dataset, prior: gpjax.gps.Prior, key) -> gpjax.gps.AbstractPosterior:
    # Our function is noise-free, so we set the observation noise's standard deviation to a very small value
    likelihood = gpjax.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=Static(jnp.array(1e-6)))
    posterior = prior * likelihood
    opt_posterior, _ = gpjax.fit(model=posterior,
                                 objective=lambda p, d: -gpjax.objectives.conjugate_mll(p, d),
                                 train_data=data,
                                 optim=optax.adam(learning_rate=0.01),
                                 num_iters=1000,
                                 safe=True,
                                 key=key,
                                 verbose=False)

    return opt_posterior

def sample_and_optimize_posterior(optimized_posteriors, D, key, lower_bound, upper_bound, num_samples, num_initial_sample_points):
    def apply_fn(sample_func, x_data):
        return sample_func(x_data)

    def create_path(xs, ys):
        return ExPath(jnp.squeeze(xs, axis=-2), jnp.squeeze(ys, axis=-2))

    key, _key = jr.split(key)
    initial_sample_points = jr.uniform(_key, shape=(num_initial_sample_points, lower_bound.shape[0]), dtype=jnp.float64,
                                       minval=lower_bound, maxval=upper_bound)  # TODO can we batch this?

    def outer_loop(key):
        sample_func_list = []
        for gp_idx, posterior in enumerate(optimized_posteriors):
            data = gpjax.Dataset(X=D.X, y=jnp.expand_dims(D.y[:, gp_idx], axis=-1))
            key, _key = jrandom.split(key)
            sample_func = posterior.sample_approx(num_samples=1, train_data=data, key=_key, num_features=500)
            sample_func_list.append(sample_func)

        def _step_fn(this_x, _):
            predictions = []
            for sample_func in sample_func_list:
                y = apply_fn(sample_func, this_x)
                predictions.append(y)

            y_tot = jnp.concatenate(predictions, axis=-1)

            next_x = jnp.clip(this_x + y_tot, jnp.array([domain[0][0], domain[1][0]]), jnp.array([domain[0][1], domain[1][1]]))

            return next_x, (this_x, y_tot)

        _, (all_xs, all_ys) = jax.lax.scan(_step_fn, init_x, None, length=40)

        exe_path = create_path(all_xs, all_ys)

        sample_mus = []
        sample_stds = []
        for gp_idx, posterior in enumerate(optimized_posteriors):
            comb_x = jnp.concatenate((D.X, exe_path.x))
            comb_y = jnp.concatenate((jnp.expand_dims(D.y[:, gp_idx], axis=-1), jnp.expand_dims(exe_path.y[:, gp_idx], axis=-1)))

            data = gpjax.Dataset(X=comb_x, y=comb_y)
            latent_dist = posterior.predict(initial_sample_points, train_data=data)
            predictive_dist = posterior.likelihood(latent_dist)

            sample_mus.append(predictive_dist.mean())  # TODO should this be predictive or should it be just posterior, if latter then do we need the points?
            sample_stds.append(predictive_dist.stddev())

        return all_xs, all_ys, exe_path, jnp.array(sample_mus), jnp.array(sample_stds)

    # Create initial state for all samples
    init_x = jnp.array([[0.5, 0.5]])
    # init_xs = jnp.tile(init_x, (num_samples, 1, 1))

    key, _key = jrandom.split(key)
    batch_key = jrandom.split(key, num_samples)

    # for posterior in optimized_posteriors:
    #     key, _key = jrandom.split(key)
    #     # L = jnp.linalg.cholesky(posterior.likelihood.covariance())
    #     # samples = posterior.likelhood.mean() + jrandom.normal(_key, shape=(num_samples,)) @ L.T
    #     samples = posterior.likelihood().mean() + jrandom.normal(_key, shape=(num_samples,)) * posterior.likelihood.stddev()

    all_xs, all_ys, exe_path, sample_mus, sample_stds = jax.vmap(outer_loop)(batch_key)

    return exe_path, initial_sample_points, sample_mus, sample_stds

def optimise_sample(optimized_posteriors, D, initial_sample_points, sample_mus, sample_stds):
    # the acquisting function thing is here
    predictive_mus = []  # TODO can we batch this
    predictive_stds = []

    for gp_idx, posterior in enumerate(optimized_posteriors):
        data = gpjax.Dataset(X=D.X, y=jnp.expand_dims(D.y[:, gp_idx], axis=-1))
        latent_dist = posterior.predict(initial_sample_points, train_data=data)
        predictive_dist = posterior.likelihood(latent_dist)

        predictive_mean = predictive_dist.mean()  # TODO should this be predictive or should it be just posterior, if latter then do we need the points?
        predictive_std = predictive_dist.stddev()
        predictive_mus.append(predictive_mean)
        predictive_stds.append(predictive_std)

        # sample_mean, sample_std = jax.vmap(test_vmap, in_axes=(None, None, 0, 0, None, None))(D.X, jnp.expand_dims(
        #     D.y[:, gp_idx], axis=-1), exe_path_x, jnp.expand_dims(exe_path_y[:, :, gp_idx], axis=-1), posterior,
        #                                                                                       initial_sample_points)
        # sample_mus.append(sample_mean)
        # sample_stds.append(sample_std)

    def acq_exe_normal(predictive, sample):
        def entropy_given_normal_std_list(std_list):
            return jnp.log(std_list) + jnp.log(jnp.sqrt(2 * jnp.pi)) + 0.5  # TODO check if correct std or var
        h_post = jnp.sum(entropy_given_normal_std_list(jnp.array(predictive)), axis=0)
        h_sample = jnp.mean(jnp.sum(entropy_given_normal_std_list(sample), axis=1), axis=0)
        acq_exe = h_post - h_sample

        return acq_exe

    acq_list = acq_exe_normal(predictive_stds, sample_stds)

    # best_x = jnp.array([initial_sample_points[jnp.argmin(initial_sample_y)]])
    # best_x = jnp.array([initial_sample_points[jnp.argmin(acq_list)]])

    # # We want to maximise the utility function, but the optimiser performs minimisation. Since we're minimising the sample drawn, the sample is actually the negative utility function.
    # negative_utility_fn = lambda x: sample(x)[0][0]
    # lbfgsb = ScipyBoundedMinimize(fun=negative_utility_fn, method="l-bfgs-b")
    # bounds = (lower_bound, upper_bound)
    # x_star = lbfgsb.run(best_x, bounds=bounds).params

    acq_idx = np.argmax(acq_list)
    acq_opt = initial_sample_points[acq_idx]
    acq_val = acq_list[acq_idx]

    return jnp.expand_dims(acq_opt, axis=0)  # x_star

lower_bound = jnp.array([domain[0][0], domain[1][0]])
upper_bound = jnp.array([domain[0][1], domain[1][1]])
n_init_data = 1
bo_iters = 25
num_samples = 20
num_initial_sample_points = 1000

init_x = jnp.array(unif_random_sample_domain(domain, n_init_data))
init_y = jnp.array([step_northwest(xi) for xi in init_x])

D = gpjax.Dataset(X=init_x, y=init_y)
output_dims = 2  # TODO can change this for later
gp_models = create_gp_models(output_dims)

start_time = time.time()
for i in range(bo_iters):
    print("---" * 5 + f" Start iteration i={i} " + "---" * 5)
    # Generate optimised posterior
    optimized_posteriors = []
    for gp_idx in range(output_dims):
        key, _key = jrandom.split(key)
        D_dim = gpjax.Dataset(X=D.X, y=jnp.expand_dims(D.y[:, gp_idx], axis=-1))
        opt_posterior = return_optimised_posterior(D_dim, gp_models[gp_idx], _key)
        optimized_posteriors.append(opt_posterior)

    # Sample from posteriors and find minimizer
    key, _key = jr.split(key)
    ind_time = time.time()
    exe_path, initial_sample_points, sample_mus, sample_stds = sample_and_optimize_posterior(optimized_posteriors, D, _key, lower_bound, upper_bound, num_samples, num_initial_sample_points)  # TODO is it potentialy quicker and more efficient to just sample from the posterior likelihood and vmap across this?
    print(f"{time.time() - ind_time} - Exe path time taken")
    ind_time = time.time()
    x_star = optimise_sample(optimized_posteriors, D, initial_sample_points, sample_mus, sample_stds)
    print(f"{time.time() - ind_time} - Optimisation time taken")
    y_star = f([x_star[0], x_star[1]])
    print(f"BO Iteration: {i + 1}, Queried Point: {x_star}, Black-Box Function Value:" f" {y_star}")

    D = D + gpjax.Dataset(X=x_star, y=jnp.expand_dims(jnp.array(y_star), axis=0))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Plot true path and posterior path samples
    plot_path_2d(true_path, ax, true_path=True)
    for path in exe_path.x:
        plot_path_2d_jax(path, ax)

    # Plot observations
    x_obs = [xi[0] for xi in D.X]
    y_obs = [xi[1] for xi in D.X]
    ax.scatter(x_obs, y_obs, color="green", s=120)

    ax.scatter(x_star[1][0], x_star[1][1], color="deeppink", s=120, zorder=100)
    ax.set(xlim=(domain[0][0], domain[0][1]), ylim=(domain[1][0], domain[1][1]), xlabel="$x_1$", ylabel="$x_2$")

    save_figure = True
    if save_figure:
        neatplot.save_figure(f"bax_multi_new{i}", "png")

print(time.time() - start_time)
