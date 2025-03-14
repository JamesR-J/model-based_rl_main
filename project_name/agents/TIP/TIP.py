"""
Model predictive control (MPC) with BAX.
"""

from argparse import Namespace
import numpy as np
from math import ceil
import logging

# from project_name.util.misc_util import dict_to_namespace
# from project_name.util.control_util import compute_return, iCEM_generate_samples
# from project_name.util.domain_util import project_to_domain
from project_name.agents.agent_base import AgentBase

import jax.numpy as jnp
from project_name.agents.TIP import get_TIP_config
import jax
from functools import partial
import colorednoise
import jax.random as jrandom
from gymnax.environments import environment
from flax import struct
from project_name.utils import MPCTransition, MPCTransitionXY, MPCTransitionXYR
import gymnax
from project_name.config import get_config
from typing import Union, Tuple
from project_name.utils import update_obs_fn, update_obs_fn_teleport, get_f_mpc, get_f_mpc_teleport
from project_name.agents.MPC import MPCAgent
from project_name import dynamics_models


class TIPAgent(MPCAgent):
    """
    An algorithm for model-predictive control. Here, the queries are concatenated states
    and actions and the output of the query is the next state.  We need the reward
    function in our algorithm as well as a start state.
    """

    def __init__(self, env, env_params, config, key):
        super().__init__(env, env_params, config, key)
        self.agent_config = get_TIP_config()

        # TODO add some import from folder check thingo
        # self.dynamics_model = dynamics_models.MOGP(env, env_params, config, self.agent_config, key)
        self.dynamics_model = dynamics_models.MOGPGPJax(env, env_params, config, self.agent_config, key)

    def make_postmean_func_const_key(self):
        def _postmean_fn(x, unused1, unused2, train_state, key):
            mu = self.dynamics_model.get_post_mu_cov_samples(x, train_state, train_state["sample_key"], full_cov=False)
            return jnp.squeeze(mu, axis=0)
        return _postmean_fn

    @partial(jax.jit, static_argnums=(0, 2))
    def _optimise(self, train_state, f, exe_path_BSOPA, x_test, key):
        curr_obs_O = x_test[:self.obs_dim]
        mean = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.action_dim))  # TODO this may not be zero if there is alreayd an action sequence, should check this
        init_var_divisor = 4
        var = jnp.ones_like(mean) * ((self.env.action_space().high - self.env.action_space().low) / init_var_divisor) ** 2

        def _iter_iCEM2(iCEM2_runner_state, unused):  # TODO perhaps we can generalise this from above
            mean_S1, var_S1, prev_samples, prev_returns, key = iCEM2_runner_state
            key, _key = jrandom.split(key)
            samples_BS1 = self._iCEM_generate_samples(_key,
                                                      self.agent_config.BASE_NSAMPS,
                                                      self.agent_config.PLANNING_HORIZON,
                                                      self.agent_config.BETA,
                                                      mean_S1,
                                                      var_S1,
                                                      self.env.action_space().low,
                                                      self.env.action_space().high)

            key, _key = jrandom.split(key)
            batch_key = jrandom.split(_key, self.agent_config.BASE_NSAMPS)
            acq = jax.vmap(self._evaluate_samples, in_axes=(None, None, None, 0, None, 0))(train_state,
                                                                                           f,
                                                                                           curr_obs_O,
                                                                                           samples_BS1,
                                                                                           exe_path_BSOPA,
                                                                                           batch_key)
            # TODO ideally we could vmap f above using params

            # TODO reinstate below so that it works with jax
            # not_finites = ~jnp.isfinite(acq)
            # num_not_finite = jnp.sum(acq)
            # # if num_not_finite > 0: # TODO could turn this into a cond
            # logging.warning(f"{num_not_finite} acq function results were not finite.")
            # acq = acq.at[not_finites[:, 0], :].set(-jnp.inf)  # TODO as they do it over iCEM samples and posterior samples, they add a mean to the posterior samples
            returns_B = jnp.squeeze(acq, axis=-1)

            # do some subset thing that works with initial dummy data, can#t do a subset but giving it a shot
            samples_concat_BP1S1 = jnp.concatenate((samples_BS1, prev_samples), axis=0)
            returns_concat_BP1 = jnp.concatenate((returns_B, prev_returns))

            # rank returns and chooses the top N_ELITES as the new mean and var
            elite_idx = jnp.argsort(returns_concat_BP1)[-self.agent_config.N_ELITES:]
            elites_ISA = samples_concat_BP1S1[elite_idx, ...]
            elite_returns_I = returns_concat_BP1[elite_idx]

            mean_SA = jnp.mean(elites_ISA, axis=0)
            var_SA = jnp.var(elites_ISA, axis=0)

            return (mean_SA, var_SA, elites_ISA, elite_returns_I, key), (samples_concat_BP1S1, returns_concat_BP1)

        key, _key = jrandom.split(key)
        init_samples = jnp.zeros((self.agent_config.N_ELITES, self.agent_config.PLANNING_HORIZON, 1))
        init_returns = jnp.ones((self.agent_config.N_ELITES,)) * -jnp.inf
        _, (tree_samples, tree_returns) = jax.lax.scan(_iter_iCEM2, (mean, var, init_samples, init_returns, _key), None, self.agent_config.OPTIMISATION_ITERS)

        flattened_samples = tree_samples.reshape(tree_samples.shape[0] * tree_samples.shape[1], -1)
        flattened_returns = tree_returns.reshape(tree_returns.shape[0] * tree_returns.shape[1], -1)

        best_idx = jnp.argmax(flattened_returns)
        best_return = flattened_returns[best_idx]
        best_sample = flattened_samples[best_idx, ...]

        optimum = jnp.concatenate((curr_obs_O, jnp.expand_dims(best_sample[0], axis=0)))

        return optimum, best_return

    @partial(jax.jit, static_argnums=(0, 2))
    def _evaluate_samples(self, train_state, f, obs_O, samples_S1, exe_path_BSOPA, key):
        # run a for loop planning basically
        def _run_planning_horizon2(runner_state, actions_A):  # TODO again can we generalise this from above to save rewriting things
            obs_O, key = runner_state
            obsacts_OPA = jnp.concatenate((obs_O, actions_A), axis=-1)
            key, _key = jrandom.split(key)
            data_y_O = f(jnp.expand_dims(obsacts_OPA, axis=0), None, None, train_state, _key)
            nobs_O = self._update_fn(obsacts_OPA, data_y_O, self.env, self.env_params)
            return (nobs_O, key), obsacts_OPA

        _, x_list_SOPA = jax.lax.scan(jax.jit(_run_planning_horizon2), (obs_O, key), samples_S1)

        # TODO this part is the acquisition function so should be generalised at some point rather than putting it here
        # get posterior covariance for x_set
        _, post_cov = self.dynamics_model.get_post_mu_full_cov(x_list_SOPA, train_state, full_cov=True)

        # get posterior covariance for all exe_paths, so this be a vmap probably
        def _get_sample_cov(x_list_SOPA, exe_path_SOPA, params):
            params["train_data_x"] = jnp.concatenate((params["train_data_x"], exe_path_SOPA["exe_path_x"]))
            params["train_data_y"] = jnp.concatenate((params["train_data_y"], exe_path_SOPA["exe_path_y"]))
            return self.dynamics_model.get_post_mu_full_cov(x_list_SOPA, params, full_cov=True)
        # TODO this is fairly slow as it feeds in a large amount of gp data to get the sample cov
        # TODO can we speed this up?

        _, samp_cov = jax.vmap(_get_sample_cov, in_axes=(None, 0, None))(x_list_SOPA, exe_path_BSOPA, train_state)

        def fast_acq_exe_normal(post_covs, samp_covs_list):
            signs, dets = jnp.linalg.slogdet(post_covs)
            h_post = jnp.sum(dets, axis=-1)
            signs, dets = jnp.linalg.slogdet(samp_covs_list)
            h_samp = jnp.sum(dets, axis=-1)
            avg_h_samp = jnp.mean(h_samp, axis=-1)
            acq_exe = h_post - avg_h_samp
            return acq_exe

        acq = fast_acq_exe_normal(jnp.expand_dims(post_cov, axis=0), samp_cov)

        return acq

    # @partial(jax.jit, static_argnums=(0,))
    def get_next_point(self, curr_obs, train_state, key):
        key, _key = jrandom.split(key)
        batch_key = jrandom.split(_key, self.agent_config.ACQUISITION_SAMPLES)

        def sample_key_train_state(train_state, key):
            train_state["sample_key"] = key
            return train_state

        batch_train_state = jax.vmap(sample_key_train_state, in_axes=(None, 0))(train_state, batch_key)
        # TODO kind of dodgy fix to get samples for the posterior but is it okay?

        # idea here is to run a batch of MPC on different posterior functions, can we sample a batch of params?
        # so that we can just call the GP on these params in a VMAPPED setting
        _, exe_path_BSOPA = jax.vmap(self.execute_mpc, in_axes=(None, None, 0, 0, None, None))(
            self.make_postmean_func_const_key(),
            # self.make_postmean_func(),
            curr_obs,
            batch_train_state,
            batch_key,
            self.env_params.horizon,
            self.agent_config.ACTIONS_PER_PLAN)

        # add in some test values
        key, _key = jrandom.split(key)
        x_test = jnp.concatenate((curr_obs, self.env.action_space(self.env_params).sample(_key)))

        # now optimise the dynamics model with the x_test
        # take the exe_path_list that has been found with different posterior samples using iCEM
        # x_data and y_data are what ever you have currently
        key, _key = jrandom.split(key)
        x_next, acq_val = self._optimise(train_state, self.make_postmean_func(), exe_path_BSOPA, x_test, _key)

        assert jnp.allclose(curr_obs, x_next[:self.obs_dim]), "For rollout cases, we can only give queries which are from the current state"
        # TODO can we make this jittable?

        return x_next, exe_path_BSOPA, curr_obs, train_state, acq_val, key
