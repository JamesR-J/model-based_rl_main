"""
PETS implementation based off "https://github.com/kchua/mbrl-jax/tree/master"
"""

from argparse import Namespace
import numpy as np
from math import ceil
import logging

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

    def create_train_state(self, init_data_x, init_data_y, key):
        return self.dynamics_model.create_train_state(init_data_x, init_data_y, key)

    def pretrain_params(self, init_data_x, init_data_y, key):
        # add some batch data call for each iteration of the loop

        train_state = self.create_train_state(init_data_x, init_data_y, key)

        def update_fn(update_state, unused):
            batch_x = jnp.reshape(init_data_x, (self.agent_config.NUM_ENSEMBLE, -1, init_data_x.shape[-1]))
            batch_y = jnp.reshape(init_data_y, (self.agent_config.NUM_ENSEMBLE, -1, init_data_y.shape[-1]))
            # TODO is the above needed for speedups?
            loss, new_update_state = self.dynamics_model.update(batch_x, batch_y, update_state)
            return new_update_state, loss

        new_train_state, init_losses = jax.lax.scan(update_fn, train_state, None, self.agent_config.NUM_INIT_UPDATES)
        # TODO do we wanna plot these initial losses?

        return new_train_state

    @partial(jax.jit, static_argnums=(0, 1, 5, 6))
    def execute_mpc(self, f, obs, train_state, key, horizon, actions_per_plan):
        key, _key = jrandom.split(key)
        batch_key = jrandom.split(_key, self.agent_config.NUM_ENSEMBLE)

        # run iCEM on each dynamics model and then select the best return is that okay? the og does it for each rollout
        # so each iCEM step has the optimal action then retries, is that better maybe?
        # the original does for samples within iCEM has diff members of the ensemble
        full_path_USX, output_USX, sample_returns_USB = jax.vmap(self.run_algorithm_on_f, in_axes=(None, None, 0, 0, None, None))(f, obs, train_state, batch_key, horizon, actions_per_plan)
        # above basically runs the optimal iCEM for each dynamics model and then we find the best one, is that better or worse than the original method?

        # given the returns we find the optimal one over the batch for each ensemble
        best_batch_sample_returns_idx = jnp.argmax(sample_returns_USB)
        best_sample_returns_idx_USB = jnp.unravel_index(best_batch_sample_returns_idx, sample_returns_USB.shape)
        output_SX = jax.tree_util.tree_map(lambda x: x[best_sample_returns_idx_USB[0], ...], output_USX)

        action_SA = output_SX[1]

        exe_path_USX = self.get_exe_path_crop(output_USX[0], output_USX[1])
        # outputs batch of exe_path rather than just one

        return action_SA, exe_path_USX, output_SX

    def make_postmean_func(self):
        def _postmean_fn(x, env, unused2, train_state, key):
            key, _key = jrandom.split(key)
            # ensemble_idx = jax.random.randint(train_state, minval=0, maxval=self.agent_config.NUM_ENSEMBLE, shape=())
            # ensemble_params = jax.tree_util.tree_map(lambda x: x[ensemble_idx], train_state)
            mu, std = self.dynamics_model.predict(x, train_state, _key)
            # return jnp.squeeze(mu, axis=0)  # TODO in original it is obs + mu, check this
            return jnp.squeeze(x[..., :env.obs_dim] + mu, axis=0)
        return _postmean_fn

    def make_postmean_func2(self):
        def _postmean_fn(x, unused1, unused2, train_state, key):
            key, _key = jrandom.split(key)
            ind_train_state = jax.tree_util.tree_map(lambda x: x[0], train_state)
            # TODO a dodgy fix for now setting the 0th ensemble member
            mu, std = self.dynamics_model.predict(x, ind_train_state, _key)
            return mu
        return _postmean_fn

    @partial(jax.jit, static_argnums=(0, 2))
    def _optimise(self, train_state, f, exe_path_BSOPA, x_test, key):
        # curr_obs_O = x_test[:self.obs_dim]
        # mean = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.action_dim))  # TODO this may not be zero if there is alreayd an action sequence, should check this
        # init_var_divisor = 4
        # var = jnp.ones_like(mean) * ((self.env.action_space().high - self.env.action_space().low) / init_var_divisor) ** 2
        #
        # def _iter_iCEM2(iCEM2_runner_state, unused):  # TODO perhaps we can generalise this from above
        #     mean_S1, var_S1, prev_samples, prev_returns, key = iCEM2_runner_state
        #     key, _key = jrandom.split(key)
        #     samples_BS1 = self._iCEM_generate_samples(_key,
        #                                               self.agent_config.BASE_NSAMPS,
        #                                               self.agent_config.PLANNING_HORIZON,
        #                                               self.agent_config.BETA,
        #                                               mean_S1,
        #                                               var_S1,
        #                                               self.env.action_space().low,
        #                                               self.env.action_space().high)
        #
        #     key, _key = jrandom.split(key)
        #     batch_key = jrandom.split(_key, self.agent_config.BASE_NSAMPS)
        #     acq = jax.vmap(self._evaluate_samples, in_axes=(None, None, None, 0, None, 0))(train_state,
        #                                                                                    f,
        #                                                                                    curr_obs_O,
        #                                                                                    samples_BS1,
        #                                                                                    exe_path_BSOPA,
        #                                                                                    batch_key)
        #
        #     # TODO reinstate below so that it works with jax
        #     # not_finites = ~jnp.isfinite(acq)
        #     # num_not_finite = jnp.sum(acq)
        #     # # if num_not_finite > 0: # TODO could turn this into a cond
        #     # logging.warning(f"{num_not_finite} acq function results were not finite.")
        #     # acq = acq.at[not_finites[:, 0], :].set(-jnp.inf)  # TODO as they do it over iCEM samples and posterior samples, they add a mean to the posterior samples
        #     returns_B = jnp.squeeze(acq, axis=-1)
        #
        #     # do some subset thing that works with initial dummy data, can#t do a subset but giving it a shot
        #     samples_concat_BP1S1 = jnp.concatenate((samples_BS1, prev_samples), axis=0)
        #     returns_concat_BP1 = jnp.concatenate((returns_B, prev_returns))
        #
        #     # rank returns and chooses the top N_ELITES as the new mean and var
        #     elite_idx = jnp.argsort(returns_concat_BP1)[-self.agent_config.N_ELITES:]
        #     elites_ISA = samples_concat_BP1S1[elite_idx, ...]
        #     elite_returns_I = returns_concat_BP1[elite_idx]
        #
        #     mean_SA = jnp.mean(elites_ISA, axis=0)
        #     var_SA = jnp.var(elites_ISA, axis=0)
        #
        #     return (mean_SA, var_SA, elites_ISA, elite_returns_I, key), (samples_concat_BP1S1, returns_concat_BP1)
        #
        # key, _key = jrandom.split(key)
        # init_samples = jnp.zeros((self.agent_config.N_ELITES, self.agent_config.PLANNING_HORIZON, 1))
        # init_returns = jnp.ones((self.agent_config.N_ELITES,)) * -jnp.inf
        # _, (tree_samples, tree_returns) = jax.lax.scan(_iter_iCEM2, (mean, var, init_samples, init_returns, _key), None, self.agent_config.OPTIMISATION_ITERS)
        #
        # flattened_samples = tree_samples.reshape(tree_samples.shape[0] * tree_samples.shape[1], -1)
        # flattened_returns = tree_returns.reshape(tree_returns.shape[0] * tree_returns.shape[1], -1)
        #
        # best_idx = jnp.argmax(flattened_returns)
        # best_return = flattened_returns[best_idx]
        # best_sample = flattened_samples[best_idx, ...]
        #
        # optimum = jnp.concatenate((curr_obs_O, jnp.expand_dims(best_sample[0], axis=0)))

        return train_state

    # @partial(jax.jit, static_argnums=(0,))
    def get_next_point(self, curr_obs_O, train_state, key):
        action_1A, exe_path_USOPA, _ = self.execute_mpc(self.make_postmean_func(),
                                                                          curr_obs_O,
                                                                          train_state,
                                                                          key,
                                                                          1, 1)

        x_next_OPA = jnp.concatenate((curr_obs_O, jnp.squeeze(action_1A, axis=0)))

        # add in some test values
        key, _key = jrandom.split(key)
        x_test = jnp.concatenate((curr_obs_O, self.env.action_space(self.env_params).sample(_key)))
        # TODO do we need to use a test set or something adjacent?

        key, _key = jrandom.split(key)
        train_state = self._optimise(train_state, self.make_postmean_func(), exe_path_USOPA, x_test, _key)
        # TODO do I need the whole paths from execute_mpc to optimise or can I just have the usual dataset?

        assert jnp.allclose(curr_obs_O, x_next_OPA[:self.obs_dim]), "For rollout cases, we can only give queries which are from the current state"
        # TODO can we jax the assertion?

        return x_next_OPA, exe_path_USOPA, curr_obs_O, train_state, None, key

# TODO could add some testing in