"""
PETS implementation based off "https://github.com/kchua/mbrl-jax/tree/master"
"""

from argparse import Namespace
import numpy as np
from math import ceil
import logging

from project_name.util.misc_util import dict_to_namespace
from project_name.util.control_util import compute_return, iCEM_generate_samples
from project_name.util.domain_util import project_to_domain
from project_name.agents.agent_base import AgentBase

import jax.numpy as jnp
from project_name.agents.PETS import get_PETS_config
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


class PETSAgent(MPCAgent):
    """
    Just uses ensemble of NN as a dynamics model and runs out an MPC plan using (i)CEM
    """

    def __init__(self, env, env_params, config, key):
        super().__init__(env, env_params, config, key)
        self.agent_config = get_PETS_config()

        self.dynamics_model = dynamics_models.NeuralNetDynamicsModel(env, env_params, config, self.agent_config, key)

    def make_postmean_func(self):
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

    @partial(jax.jit, static_argnums=(0,))
    def get_next_point(self, curr_obs, train_state, key):
        key, _key = jrandom.split(key)
        batch_key = jrandom.split(_key, self.agent_config.NUM_ENSEMBLE)

        # idea here is to run a batch of MPC on different posterior functions, can we sample a batch of params?
        # so that we can just call the GP on these params in a VMAPPED setting
        action, exe_path_BSOPA = jax.vmap(self.execute_mpc, in_axes=(None, None, 0, 0, None, None))(
            self.make_postmean_func(),
            curr_obs,
            train_state,
            batch_key,
            self.env_params.horizon,
            self.agent_config.ACTIONS_PER_PLAN)
        x_next = jnp.concatenate((jnp.expand_dims(curr_obs, axis=0), action), axis=-1)

        key, _key = jrandom.split(key)
        train_state = self._optimise(train_state, _key)

        return jnp.squeeze(x_next, axis=0), exe_path_BSOPA, curr_obs, train_state, None, key

# TODO could add some testing in