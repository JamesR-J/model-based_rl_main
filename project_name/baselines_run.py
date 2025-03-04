import jax.numpy as jnp
import jax
import jax.random as jrandom
from project_name.config import get_config  # TODO dodge need to know how to fix this
import wandb
from typing import NamedTuple
import chex
from project_name.agents import Agent
from project_name.envs.wrappers import NormalizedEnv, make_normalized_reward_function
from project_name.utils import Transition, EvalTransition
import sys
import gymnasium as gym
from project_name import envs
import logging
import gymnax
from gymnax.environments import environment
from flax import struct
from project_name import dynamics_models
from project_name.envs.gymnax_pilco_cartpole import GymnaxPilcoCartPole  # TODO add some register thing here instead
from functools import partial
import time


@struct.dataclass  # TODO dodgy for now and need to change the gymnax envs to be better for this
class EnvState(environment.EnvState):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int


def run_train(config):
    key = jrandom.PRNGKey(config.SEED)

    # TODO add the plot functionality as required

    # init the environment and reward function specific to it, unsure if still need this reward function thing?
    # env = gym.make(config.ENV_NAME)
    # reward_function = envs.reward_functions[config.ENV_NAME]  # TODO add the custom reward function thing

    env = GymnaxPilcoCartPole()
    env_params = env.default_params
    obs_dim = len(env.observation_space(env_params).low)  # TODO is there a better way to write this?
    action_dim = env.action_space().shape[0]  # TODO same for this

    # normalise env if required as well as reward function
    # if config.NORMALISE_ENV:  # TODO add this for gymnax etc
    #     env = NormalizedEnv(env)
    #     if reward_function is not None:
    #         reward_function = make_normalized_reward_function(env, reward_function)
    #     # plot_fn = make_normalized_plot_fn(env, plot_fn)  # TODO add plot normalisation aswell

    # TODO move the below somewhere better
    @partial(jax.jit, static_argnums=(2,))
    def _get_f_mpc(x_OPA, key, use_info_delta=False):  # TODO this should be generalised out of class at some point
        obs_O = x_OPA[:obs_dim]
        action_A = x_OPA[obs_dim:]
        env_state = EnvState(x=obs_O[0], x_dot=obs_O[1], theta=obs_O[2], theta_dot=obs_O[3], time=0)  # TODO specific for cartpole, need to generalise this
        nobs_O, _, _, _, info = env.step(key, env_state, action_A, env_params)
        return nobs_O - obs_O

    # set the initial obs, i.e. env.reset
    def get_start_obs(key, obs=None):  # TODO some if statement if have some fixed start point
        key, _key = jrandom.split(key)
        obs, env_state = env.reset(_key)
        logging.info(f"Start obs: {obs}")
        return obs, env_state
    start_obs, start_env_state = get_start_obs(key)

    # add the planner/actor
    key, _key = jrandom.split(key)
    actor = Agent(env=env, env_params=env_params, config=config, utils=None, key=_key)

    # add the dynamics model # TODO perhaps make this like part of a specific actor rather than separate here?
    key, _key = jrandom.split(key)
    # dynamics_model = dynamics_model.NeuralNetDynamicsModel(init_obs, env.action_space().sample(_key), hidden_dims=[50, 50],
    #                                         hidden_activations=jax.nn.swish, is_probabilistic=True)
    dynamics_model = dynamics_models.MOGP(env, env_params, config, None, key)
    dynamics_model_train_state = dynamics_model.create_train_state(start_obs)  # TODO kinda dodge but adds the first obs to the GP dataset as otherwise have an empty params that would not work I think?
    # TODO as mentioned above we add obs and some arbitrary action, but this may impact the model greatly so should fix this
    # TODO generalise this and add some more dynamics models

    def get_initial_data(config, env, f, plot_fn, key):
        def unif_random_sample_domain(low, high, key, n=1):
            unscaled_random_sample = jrandom.uniform(key, shape=(n, low.shape[0]))
            scaled_random_sample = low[..., :] + (high[..., :] - low[..., :]) * unscaled_random_sample
            return scaled_random_sample

        low = jnp.concatenate((env.observation_space(env_params).low, jnp.expand_dims(jnp.array(env.action_space().low,), axis=0)))
        high = jnp.concatenate((env.observation_space(env_params).high, jnp.expand_dims(jnp.array(env.action_space().high,), axis=0)))

        data_x_LOPA = unif_random_sample_domain(low, high, key, n=config.NUM_INIT_DATA)
        if config.GENERATIVE_ENV:
            batch_key = jrandom.split(key, config.NUM_INIT_DATA)
            data_y_LO = f(data_x_LOPA, batch_key)
        else:
            raise NotImplementedError("If not generative env then we have to output nothing, unsure how to do in Jax")

        # # Plot initial data  # TODO add data plotting
        # ax_obs_init, fig_obs_init = plot_fn(path=None, domain=domain)
        # if ax_obs_init is not None and config.save_figures:
        #     plot(ax_obs_init, data.x, "o", color="k", ms=1)
        #     fig_obs_init.suptitle("Initial Observations")
        #     neatplot.save_figure(str(dumper.expdir / "obs_init"), "png", fig=fig_obs_init)

        return data_x_LOPA, data_y_LO

    # get some initial data for something, unsure what as of now? also a test set
    key, _key = jrandom.split(key)
    init_data_x, init_data_y = get_initial_data(config, env, jax.vmap(_get_f_mpc), None, _key)  # TODO add plot function
    # key, _key = jrandom.split(key)
    # test_data_x, test_data_y = get_initial_data(config, env, f, None, _key)  # TODO add plot function

    @partial(jax.jit, static_argnums=(1,))
    def execute_gt_mpc(init_obs, f, key):
        key, _key = jrandom.split(key)
        full_path, test_mpc_data, all_returns = actor.agent.run_algorithm_on_f(f, init_obs, _key,
                                                                  horizon=config.ENV_HORIZON,  # TODO p sure it is this, but should come from the env and not from config?
                                                                  actions_per_plan=actor.agent.agent_config.ACTIONS_PER_PLAN)
        path_lengths = len(full_path[0])  # TODO should we turn the output into a dict for x and y ?
        true_path = actor.agent.get_exe_path_crop(test_mpc_data[0], test_mpc_data[1])

        key, _key = jrandom.split(key)
        test_points = jax.tree_util.tree_map(lambda x: jrandom.choice(_key, x,
                                                                      (config.TEST_SET_SIZE // config.NUM_EVAL_TRIALS, )), true_path)
        # TODO ensure it samples the same pairs

        return true_path, test_points, path_lengths, all_returns

    # get some groundtruth data
    batch_key = jrandom.split(key, config.NUM_EVAL_TRIALS)
    start_gt_time = time.time()
    true_path, test_points, path_lengths, all_returns = jax.vmap(execute_gt_mpc, in_axes=(None, None, 0))(start_obs, _get_f_mpc, batch_key)
    flattened_true_path = jax.tree_util.tree_map(lambda x: x.reshape((x.shape[0] * x.shape[1], -1)), true_path)
    flattened_test_points = jax.tree_util.tree_map(lambda x: x.reshape((config.TEST_SET_SIZE, -1)), test_points)
    logging.info(f"Ground truth time taken = {time.time() - start_gt_time:.2f}s; "
                 f"Mean Return = {jnp.mean(all_returns):.2f}; Std Return = {jnp.std(all_returns):.2f}; "
                 f"Mean Path Lengths = {jnp.mean(path_lengths)}; ")

    # # Plot groundtruth paths and print info  # TODO plot these gt
    # ax_gt, fig_gt = plot_fn(true_path, ax_gt, fig_gt, domain, "samp")
    # returns.append(compute_return(output[2], 1))
    # stats = {"Mean Return": np.mean(returns), "Std Return:": np.std(returns)}

    # TODO add some hyperparameter fit on the GT data or if we are evaluating the hyperparams

    def _main_loop(main_loop_runner, init_data):  # TODO run this for config.num_iters
        curr_obs_O, env_state, key = main_loop_runner
        init_data_x, init_data_y = init_data
        # log some info that we need basically
        # logging.info("---" * 5 + f" Start iteration i={i} " + "---" * 5)
        # logging.info(f"Length of data.x: {len(data.x)}")
        # logging.info(f"Length of data.y: {len(data.y)}")

        def get_next_point(curr_obs, init_data_x, init_data_y, key):
            def make_postmean_func(model):
                def _postmean_fn(x, key):
                    x = jnp.expand_dims(x, axis=0)  # TODO adding in num_data_points dim as required for GPJax
                    mu, std = model.get_post_mu_cov(x, dynamics_model_train_state, full_cov=False)
                    return jnp.squeeze(mu, axis=0)  # TODO this removes the first num_data_points dim

                return _postmean_fn

            # TODO some if statement if our input data does not exist as not using generative approach, i.e. the first step

            # TODO if statement if using an acquisition function, idk how to do this so that we don't require if statement
            # idea here is to run a batch of MPC on different functions from the posterior
            # so can we sample like a batch of params? so that we can just call the GP on these params in a VMAPPED setting

            # we need to make a posterior params list that is vmappable for future use
            key, _key = jrandom.split(key)
            batch_key = jrandom.split(_key, actor.agent.agent_config.ACQUISITION_SAMPLES)
            action_B1A, exe_path_BSOPA = jax.vmap(actor.agent.execute_mpc, in_axes=(None, None, 0))(make_postmean_func(dynamics_model),
                                                                                     curr_obs,
                                                                                     batch_key)
            exe_path = exe_path_BSOPA  # TODO not sure what to do with this but be good for plotting I guess
            # TODO maybe some postmena function that takes specific keys dependant on the vamp batch

            # add in some test values
            key, _key = jrandom.split(key)
            x_test = jnp.concatenate((curr_obs, env.action_space().sample(_key)))

            #  now optimise the dynamics model with the x_test
            # take the exe_path_list that has been found with different posterior samples using iCEM
            # x_data and y_data are what ever you have currently
            # x_list is creating a trajectory from the iCEM samples using the posterior samples, how is this diff to exe path?
            key, _key = jrandom.split(key)
            x_next, acq_val = actor.agent.optimise(dynamics_model, dynamics_model_train_state, make_postmean_func(dynamics_model), exe_path_BSOPA, x_test, _key)
            # TODO figure out how we can batch this dynamics model etc that would be the easiest


            # samples = dynamics_model.return_posterior_samples(
            #     jnp.concatenate((jnp.expand_dims(curr_obs, axis=0), jnp.zeros((1, 1))), axis=-1),
            #     dynamics_model_train_state, _key)

            # If using MPC then
            # TODO some conditional for MPC here and for acquisition stuff above
            # key, _key = jrandom.split(key)
            # action, exe_path = actor.agent.execute_mpc(make_postmean_func(dynamics_model), curr_obs, key)
            # x_next = jnp.concatenate((jnp.expand_dims(curr_obs, axis=0), action), axis=-1)
            # x_next = jnp.squeeze(x_next, axis=0)

            return x_next, exe_path

        # get next point
        x_next_OPA, exe_path = get_next_point(curr_obs_O, init_data_x, init_data_y, key)

        # periodically run evaluation and plot

        # Query function, update data
        key, _key = jrandom.split(key)
        if config.GENERATIVE_ENV & config.NO_ENV_RUN:
            y_next = _get_f_mpc(x_next_OPA, _key)
            nobs_O = y_next + curr_obs_O
            new_env_state = "UHOH"
        else:
            action_A = x_next_OPA[-action_dim:]
            nobs_O, new_env_state, reward, done, info = env.step(_key, env_state, action_A, env_params)
            y_next = nobs_O - curr_obs_O

        # TODO somehow update the dataset that is used in the GP but also generally in the loop

        new_x = jnp.concatenate((init_data_x, jnp.expand_dims(x_next_OPA, axis=0)))
        new_y = jnp.concatenate((init_data_y, jnp.expand_dims(y_next, axis=0)))

        return (nobs_O, new_env_state, key), (new_x, new_y)

    key, _key = jrandom.split(key)
    init_x = jnp.expand_dims(jnp.concatenate((start_obs, env.action_space().sample(_key)), axis=-1), axis=0)
    init_y = jnp.zeros((1, obs_dim))
    stuff = jax.lax.scan(_main_loop, (start_obs, start_env_state, _key), (init_x, init_y), 20)

    return


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=True):
        train = run_train(config)
