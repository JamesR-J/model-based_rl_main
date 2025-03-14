import gpjax as gpx
import GPJax_AScannell as gpjaxas
from dynamics_models import MOGP
from typing import (
    List,
    Tuple,
)

import jax

# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy as np
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

from gpjax.parameters import Static
from gpjax.typing import (
    Array,
    FunctionalSample,
    ScalarFloat,
)

config.update("jax_enable_x64", True)

key = jr.key(42)

cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


# def standardised_six_hump_camel(x: Float[Array, "N 2"]) -> Float[Array, "N 1"]:
#     mean = 1.12767
#     std = 1.17500
#     x1 = x[..., :1]
#     x2 = x[..., 1:]
#     term1 = (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
#     term2 = x1 * x2
#     term3 = (-4 + 4 * x2**2) * x2**2
#     return (term1 + term2 + term3 - mean) / std
#
#
# def return_optimised_posterior(
#     data: gpx.Dataset, prior: gpx.gps.Prior, key: Array) -> gpx.gps.AbstractPosterior:
#     # Our function is noise-free, so we set the observation noise's standard deviation to a very small value
#     likelihood = gpx.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=Static(jnp.array(1e-6)))
#
#     posterior = prior * likelihood
#
#     opt_posterior, _ = gpx.fit(
#         model=posterior,
#         objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
#         train_data=data,
#         optim=ox.adam(learning_rate=0.01),
#         num_iters=1000,
#         safe=True,
#         key=key,
#         verbose=False,
#     )
#
#     return opt_posterior
#
#
# def optimise_sample(
#     sample: FunctionalSample,
#     key: Int[Array, ""],
#     lower_bound: Float[Array, "D"],
#     upper_bound: Float[Array, "D"],
#     num_initial_sample_points: int,
# ) -> ScalarFloat:
#     initial_sample_points = jr.uniform(
#         key,
#         shape=(num_initial_sample_points, lower_bound.shape[0]),
#         dtype=jnp.float64,
#         minval=lower_bound,
#         maxval=upper_bound,
#     )
#     initial_sample_y = sample(initial_sample_points)
#     best_x = jnp.array([initial_sample_points[jnp.argmin(initial_sample_y)]])
#
#     # We want to maximise the utility function, but the optimiser performs minimisation. Since we're minimising the sample drawn, the sample is actually the negative utility function.
#     negative_utility_fn = lambda x: sample(x)[0][0]
#     lbfgsb = ScipyBoundedMinimize(fun=negative_utility_fn, method="l-bfgs-b")
#     bounds = (lower_bound, upper_bound)
#     x_star = lbfgsb.run(best_x, bounds=bounds).params
#     return x_star
#
#
# lower_bound = jnp.array([-2.0, -1.0])
# upper_bound = jnp.array([2.0, 1.0])
# initial_sample_num = 5
# bo_iters = 12
# num_experiments = 5
# bo_experiment_results = []
#
# for experiment in range(num_experiments):
#     print(f"Starting Experiment: {experiment + 1}")
#     # Set up initial dataset
#     initial_x = tfp.mcmc.sample_halton_sequence(dim=2, num_results=initial_sample_num, seed=key, dtype=jnp.float64)
#     initial_x = jnp.array(lower_bound + (upper_bound - lower_bound) * initial_x)
#     initial_y = standardised_six_hump_camel(initial_x)
#     D = gpx.Dataset(X=initial_x, y=initial_y)
#
#     for i in range(bo_iters):
#         key, subkey = jr.split(key)
#
#         # Generate optimised posterior
#         mean = gpx.mean_functions.Zero()
#         kernel = gpx.kernels.Matern52(active_dims=[0, 1], lengthscale=jnp.array([1.0, 1.0]), variance=2.0)
#         prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
#         opt_posterior = return_optimised_posterior(D, prior, subkey)
#
#         # Draw a sample from the posterior, and find the minimiser of it
#         approx_sample = opt_posterior.sample_approx(num_samples=1, train_data=D, key=subkey, num_features=500)
#         x_star = optimise_sample(approx_sample, subkey, lower_bound, upper_bound, num_initial_sample_points=1000)
#
#         # Evaluate the black-box function at the best point observed so far, and add it to the dataset
#         y_star = standardised_six_hump_camel(x_star)
#         print(f"BO Iteration: {i + 1}, Queried Point: {x_star}, Black-Box Function Value: {y_star}")
#         D = D + gpx.Dataset(X=x_star, y=y_star)
#     bo_experiment_results.append(D)


# Set the random seed for reproducibility
key = jr.PRNGKey(42)


# Multi-output version of the standardized six-hump camel function
def multi_output_six_hump_camel(x: Float[Array, "N 2"]) -> Float[Array, "N 2"]:
    """
    A two-output version of the six-hump camel function.

    Args:
        x: Input array of shape (N, 2)

    Returns:
        Array of shape (N, 2) containing two outputs for each input
    """
    x1 = x[..., :1]
    x2 = x[..., 1:]

    # First output: Standard six-hump camel
    term1 = (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    y1 = term1 + term2 + term3

    # Second output: Modified version with different dynamics
    term4 = (1 - x1 ** 2 + x1 ** 4 / 20) * x1 ** 2
    term5 = x1 * x2 / 2
    term6 = (2 + 2 * x2 ** 2) * x2 ** 2
    y2 = term4 + term5 + term6

    # Standardization parameters
    mean1, std1 = 1.12767, 1.17500
    mean2, std2 = 2.58432, 1.78941

    # Standardize both outputs
    standardized_y1 = (y1 - mean1) / std1
    standardized_y2 = (y2 - mean2) / std2

    # Combine into a single array
    return jnp.concatenate([standardized_y1, standardized_y2], axis=1)


# For compatibility with single output code
def standardised_six_hump_camel(x: Float[Array, "N 2"]) -> Float[Array, "N 1"]:
    """Single output version for backward compatibility"""
    mean = 1.12767
    std = 1.17500
    x1 = x[..., :1]
    x2 = x[..., 1:]
    term1 = (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    return (term1 + term2 + term3 - mean) / std


def plot_function_surface(func, lower_bound, upper_bound, output_idx=0, title="Function Surface"):
    """
    Creates a 3D surface plot of the target function.

    Args:
        func: The function to plot
        lower_bound: Lower bounds for x1 and x2
        upper_bound: Upper bounds for x1 and x2
        output_idx: Index of the output to plot (for multi-output functions)
        title: Plot title
    """
    # Create a grid of points
    x1 = np.linspace(lower_bound[0], upper_bound[0], 100)
    x2 = np.linspace(lower_bound[1], upper_bound[1], 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Evaluate the function at these points
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = jnp.array([[X1[i, j], X2[i, j]]])
            result = func(x)
            if result.shape[1] > 1:  # Multi-output case
                Z[i, j] = result[0, output_idx]
            else:  # Single-output case
                Z[i, j] = result[0, 0]

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8)

    # Add labels and colorbar
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Function Value')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig, ax


def plot_bo_progress(dataset, func, lower_bound, upper_bound, output_idx=0, title="BO Progress"):
    """
    Plots the progress of the Bayesian optimization.

    Args:
        dataset: GPJax dataset containing the sampled points
        func: The target function
        lower_bound: Lower bounds for x1 and x2
        upper_bound: Upper bounds for x1 and x2
        output_idx: Index of the output to plot
        title: Plot title
    """
    # Create surface plot
    fig, ax = plot_function_surface(func, lower_bound, upper_bound, output_idx, title)

    # Extract points from dataset
    X = dataset.X.copy()

    # Extract corresponding output values
    if func(X).shape[1] > 1:  # Multi-output case
        y = jnp.array([func(X[i:i + 1])[0, output_idx] for i in range(X.shape[0])])
    else:  # Single-output case
        y = dataset.y.copy()
        y = y.reshape(-1)

    # Plot the sampled points
    # Color points by iteration order (from blue to red)
    colors = cm.jet(jnp.linspace(0, 1, len(X)))
    ax.scatter(X[:, 0], X[:, 1], y, c=colors, s=50, marker='o', edgecolors='k')

    # Add a colorbar for the iteration order
    sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=0, vmax=len(X)))
    sm.set_array([])
    cbar = fig.colorbar(sm, shrink=0.5, aspect=5)
    cbar.set_label('Iteration')

    return fig


def visualize_pareto_front(datasets, func):
    """
    Visualizes the Pareto front for multi-objective optimization.

    Args:
        datasets: List of GPJax datasets from multiple experiments
        func: The multi-output target function
    """
    plt.figure(figsize=(10, 8))

    for i, dataset in enumerate(datasets):
        X = dataset.X.copy()
        # Get both outputs for each point
        Y = func(X)

        # Plot all points
        plt.scatter(Y[:, 0], Y[:, 1], label=f'Experiment {i + 1}', alpha=0.6)

    # Add labels and legend
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Objective Space and Approximate Pareto Front')
    plt.legend()
    plt.grid(True)

    return plt.gcf()


def return_optimised_posterior(data: gpx.Dataset, prior: gpx.gps.Prior, key: Array, output_idx: int = 0) -> gpx.gps.AbstractPosterior:
    """
    Returns an optimized posterior based on the provided data and prior.

    Args:
        data: The dataset to fit to
        prior: The GP prior
        key: JAX random key
        output_idx: Index of the output to model (for multi-output case)

    Returns:
        Optimized posterior
    """
    # Extract the appropriate output dimension if using multi-output function
    if data.y.shape[1] > 1:
        # Create a new dataset with just the selected output
        data = gpx.Dataset(X=data.X, y=data.y[:, output_idx:output_idx + 1])

    # Our function is noise-free, so we set the observation noise's standard deviation to a very small value
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=Static(jnp.array(1e-6)))

    posterior = prior * likelihood

    opt_posterior, _ = gpx.fit(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=data,
        optim=ox.adam(learning_rate=0.01),
        num_iters=1000,
        safe=True,
        key=key,
        verbose=False,
    )

    return opt_posterior


def optimise_sample(
        sample: FunctionalSample,
        key: Int[Array, ""],
        lower_bound: Float[Array, "D"],
        upper_bound: Float[Array, "D"],
        num_initial_sample_points: int,
) -> ScalarFloat:
    """
    Optimizes a sample function drawn from a GP posterior.

    Args:
        sample: Function sample from GP posterior
        key: JAX random key
        lower_bound: Lower bounds for optimization
        upper_bound: Upper bounds for optimization
        num_initial_sample_points: Number of initial points for optimization

    Returns:
        Optimized parameter values
    """
    initial_sample_points = jr.uniform(
        key,
        shape=(num_initial_sample_points, lower_bound.shape[0]),
        dtype=jnp.float64,
        minval=lower_bound,
        maxval=upper_bound,
    )
    initial_sample_y = sample(initial_sample_points)
    best_x = jnp.array([initial_sample_points[jnp.argmin(initial_sample_y)]])

    # We want to maximise the utility function, but the optimiser performs minimisation. Since we're minimising the sample drawn, the sample is actually the negative utility function.
    negative_utility_fn = lambda x: sample(x)[0][0]
    lbfgsb = ScipyBoundedMinimize(fun=negative_utility_fn, method="l-bfgs-b")
    bounds = (lower_bound, upper_bound)
    x_star = lbfgsb.run(best_x, bounds=bounds).params
    return x_star


def run_multi_output_bo_experiment(
        target_function,
        lower_bound,
        upper_bound,
        initial_sample_num=5,
        bo_iters=12,
        output_idx=0,
        key=None
):
    """
    Runs a Bayesian optimization experiment for a single output of a possibly multi-output function.

    Args:
        target_function: The function to optimize
        lower_bound: Lower bounds for parameters
        upper_bound: Upper bounds for parameters
        initial_sample_num: Number of initial samples
        bo_iters: Number of BO iterations
        output_idx: Which output to optimize
        key: JAX random key

    Returns:
        Dataset containing all evaluated points
    """
    if key is None:
        key = jr.PRNGKey(0)

    # Set up initial dataset
    initial_x = tfp.mcmc.sample_halton_sequence(dim=2, num_results=initial_sample_num, seed=key, dtype=jnp.float64)
    initial_x = jnp.array(lower_bound + (upper_bound - lower_bound) * initial_x)
    initial_y = target_function(initial_x)

    # Create dataset
    D = gpx.Dataset(X=initial_x, y=initial_y)

    # Track the best value found
    best_values = []
    if D.y.shape[1] > 1:
        best_y = jnp.min(D.y[:, output_idx])
    else:
        best_y = jnp.min(D.y)
    best_values.append(best_y)

    for i in range(bo_iters):
        key, subkey = jr.split(key)

        # Generate optimised posterior
        mean = gpx.mean_functions.Zero()
        kernel = gpx.kernels.Matern52(active_dims=[0, 1], lengthscale=jnp.array([1.0, 1.0]), variance=2.0)
        prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
        opt_posterior = return_optimised_posterior(D, prior, subkey, output_idx)

        # Draw a sample from the posterior, and find the minimiser of it
        approx_sample = opt_posterior.sample_approx(num_samples=1, train_data=D, key=subkey, num_features=500)
        x_star = optimise_sample(approx_sample, subkey, lower_bound, upper_bound, num_initial_sample_points=1000)

        # Evaluate the black-box function at the best point observed so far
        y_star = target_function(x_star)

        # Update the best value found
        if y_star.shape[1] > 1:
            current_y = y_star[0, output_idx]
        else:
            current_y = y_star[0, 0]

        best_y = jnp.minimum(best_y, current_y)
        best_values.append(best_y)

        print(f"BO Iteration: {i + 1}, Queried Point: {x_star}, Function Value: {y_star}")

        # Add the new point to the dataset
        D = D + gpx.Dataset(X=x_star, y=y_star)

    return D, best_values


# Main execution
if __name__ == "__main__":
    # Set bounds for optimization
    lower_bound = jnp.array([-2.0, -1.0])
    upper_bound = jnp.array([2.0, 1.0])

    # Set experiment parameters
    initial_sample_num = 5
    bo_iters = 12
    num_experiments = 3

    # Create plots of the target functions
    print("Plotting function surfaces...")
    # Plot first objective
    plot_function_surface(
        multi_output_six_hump_camel,
        lower_bound,
        upper_bound,
        output_idx=0,
        title="Multi-Output Six-Hump Camel (Objective 1)"
    )
    plt.savefig("objective1_surface.png")

    # Plot second objective
    plot_function_surface(
        multi_output_six_hump_camel,
        lower_bound,
        upper_bound,
        output_idx=1,
        title="Multi-Output Six-Hump Camel (Objective 2)"
    )
    plt.savefig("objective2_surface.png")

    # Run experiments for first objective
    print("\nRunning experiments for Objective 1...")
    obj1_datasets = []
    obj1_best_values = []

    for experiment in range(num_experiments):
        print(f"\nStarting Experiment {experiment + 1} for Objective 1")
        key, subkey = jr.split(key)
        dataset, best_values = run_multi_output_bo_experiment(
            multi_output_six_hump_camel,
            lower_bound,
            upper_bound,
            initial_sample_num,
            bo_iters,
            output_idx=0,
            key=subkey
        )
        obj1_datasets.append(dataset)
        obj1_best_values.append(best_values)

        # Plot progress of this experiment
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(best_values)), best_values)
        plt.xlabel("BO Iteration")
        plt.ylabel("Best Objective Value")
        plt.title(f"Convergence Plot for Experiment {experiment + 1}, Objective 1")
        plt.grid(True)
        plt.savefig(f"obj1_exp{experiment + 1}_convergence.png")

        # Plot the BO progress in the parameter space
        bo_plot = plot_bo_progress(
            dataset,
            multi_output_six_hump_camel,
            lower_bound,
            upper_bound,
            output_idx=0,
            title=f"BO Progress for Experiment {experiment + 1}, Objective 1"
        )
        plt.savefig(f"obj1_exp{experiment + 1}_bo_progress.png")

    # Run experiments for second objective
    print("\nRunning experiments for Objective 2...")
    obj2_datasets = []
    obj2_best_values = []

    for experiment in range(num_experiments):
        print(f"\nStarting Experiment {experiment + 1} for Objective 2")
        key, subkey = jr.split(key)
        dataset, best_values = run_multi_output_bo_experiment(
            multi_output_six_hump_camel,
            lower_bound,
            upper_bound,
            initial_sample_num,
            bo_iters,
            output_idx=1,
            key=subkey
        )
        obj2_datasets.append(dataset)
        obj2_best_values.append(best_values)

        # Plot progress of this experiment
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(best_values)), best_values)
        plt.xlabel("BO Iteration")
        plt.ylabel("Best Objective Value")
        plt.title(f"Convergence Plot for Experiment {experiment + 1}, Objective 2")
        plt.grid(True)
        plt.savefig(f"obj2_exp{experiment + 1}_convergence.png")

        # Plot the BO progress in the parameter space
        bo_plot = plot_bo_progress(
            dataset,
            multi_output_six_hump_camel,
            lower_bound,
            upper_bound,
            output_idx=1,
            title=f"BO Progress for Experiment {experiment + 1}, Objective 2"
        )
        plt.savefig(f"obj2_exp{experiment + 1}_bo_progress.png")

    # Visualize multi-objective aspect - look at both objectives together
    print("\nVisualizing Pareto front approximation...")
    for i in range(num_experiments):
        pareto_plot = visualize_pareto_front([obj1_datasets[i], obj2_datasets[i]], multi_output_six_hump_camel)
        plt.savefig(f"pareto_front_exp{i + 1}.png")

    # Compare convergence across experiments
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    for i in range(num_experiments):
        plt.plot(range(len(obj1_best_values[i])), obj1_best_values[i], label=f"Experiment {i + 1}")
    plt.xlabel("BO Iteration")
    plt.ylabel("Best Value (Objective 1)")
    plt.title("Convergence Comparison for Objective 1")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for i in range(num_experiments):
        plt.plot(range(len(obj2_best_values[i])), obj2_best_values[i], label=f"Experiment {i + 1}")
    plt.xlabel("BO Iteration")
    plt.ylabel("Best Value (Objective 2)")
    plt.title("Convergence Comparison for Objective 2")
    plt.legend()
    plt.g