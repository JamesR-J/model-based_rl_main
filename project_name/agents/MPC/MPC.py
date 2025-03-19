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
from project_name.agents.MPC import get_MPC_config
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
from project_name import dynamics_models


class MPCAgent(AgentBase):
    """
    An algorithm for model-predictive control. Here, the queries are concatenated states
    and actions and the output of the query is the next state.  We need the reward
    function in our algorithm as well as a start state.
    """

    def __init__(self, env, env_params, config, key):
        super().__init__(env, env_params, config, key)
        self.agent_config = get_MPC_config()

        # TODO add some import from folder check thingo
        self.dynamics_model = dynamics_models.MOGP(env, env_params, config, self.agent_config, key)

        self.obs_dim = len(self.env.observation_space(self.env_params).low)
        self.action_dim = self.env.action_space().shape[0]
        # TODO match this to the other rl main stuff

        self.n_keep = ceil(self.agent_config.XI * self.agent_config.N_ELITES)

        if config.TELEPORT:
            self._update_fn = update_obs_fn_teleport
        else:
            self._update_fn = update_obs_fn

    def create_train_state(self, init_data_x, init_data_y, key):
        return self.dynamics_model.create_train_state(init_data_x, init_data_y, key)

    def pretrain_params(self, init_data_x, init_data_y, key):
        return self.dynamics_model.pretrain_params(init_data_x, init_data_y, key)

    @partial(jax.jit, static_argnums=(0, 3, 4, 5))  # TODO should really verify this
    def powerlaw_psd_gaussian_jax(self, key, exponent, base_nsamps, action_dim, time_horizon, fmin=0.0) -> jnp.ndarray:
                                """JAX implementation of Gaussian (1/f)**beta noise.

                                Based on the algorithm in:
                                Timmer, J. and Koenig, M.:
                                On generating power law noise.
                                Astron. Astrophys. 300, 707-710 (1995)

                                Parameters
                                ----------
                                key : jax.random.PRNGKey
                                    The random key for JAX's random number generator
                                exponent : float
                                    The power-spectrum exponent (beta) where S(f) = (1/f)**beta
                                size : int or tuple of ints
                                    The output shape. The last dimension is taken as time.
                                fmin : float, optional
                                    Low-frequency cutoff (default: 0.0)

                                Returns
                                -------
                                jnp.ndarray
                                    The generated noise samples with the specified power law spectrum
                                """
                                # BASE_NSAMPS, ACTION_DIM, HORIZON
                                # Calculate frequencies (assuming sample rate of 1)
                                f = jnp.fft.rfftfreq(time_horizon)

                                # Validate and normalize fmin
                                # if not (0 <= fmin <= 0.5):  # TODO add this in somehow
                                #     raise ValueError("fmin must be between 0 and 0.5")
                                fmin = jnp.maximum(fmin, 1.0 / time_horizon)

                                # Build scaling factors
                                s_scale = f
                                ix = jnp.sum(s_scale < fmin)
                                s_scale = jnp.where(s_scale < fmin, s_scale[ix], s_scale)
                                s_scale = s_scale ** (-exponent / 2.0)

                                # Calculate theoretical output standard deviation
                                w = s_scale[1:]
                                w = w.at[-1].multiply((1 + (time_horizon % 2)) / 2.0)  # Correct f = ±0.5
                                sigma = 2 * jnp.sqrt(jnp.sum(w ** 2)) / time_horizon

                                # Generate random components
                                key1, key2 = jrandom.split(key)
                                sr = jrandom.normal(key1, (base_nsamps, action_dim, len(f))) * s_scale
                                si = jrandom.normal(key2, (base_nsamps, action_dim, len(f))) * s_scale

                                # Handle special frequencies using lax.cond
                                def handle_even_case(args):
                                    si_, sr_ = args
                                    # Set imaginary part of Nyquist freq to 0 and multiply real part by sqrt(2)
                                    si_last = si_.at[..., -1].set(0.0)
                                    sr_last = sr_.at[..., -1].multiply(jnp.sqrt(2.0))
                                    return si_last, sr_last

                                def handle_odd_case(args):
                                    return args

                                si, sr = jax.lax.cond((time_horizon % 2) == 0, handle_even_case, handle_odd_case, (si, sr))

                                # DC component must be real
                                si = si.at[..., 0].set(0)
                                sr = sr.at[..., 0].multiply(jnp.sqrt(2.0))

                                # Combine components
                                s = sr + 1j * si

                                # Transform to time domain and normalize
                                y = jnp.fft.irfft(s, n=time_horizon, axis=-1) / sigma

                                return y

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _iCEM_generate_samples(self, key, nsamples, horizon, mean, var):
        # samples = (colorednoise.powerlaw_psd_gaussian(beta, size=(nsamps, action_dim, horizon)).transpose([0, 2, 1])
        #            * np.sqrt(var) + mean)
        samples = jnp.swapaxes(self.powerlaw_psd_gaussian_jax(key,
                                                              self.agent_config.BETA,
                                                              nsamples,
                                                              self.action_dim,
                                                              horizon),
                               1, 2)  * jnp.sqrt(var) + mean
        # TODO test this powerlaw thing
        # samples = jrandom.normal(key, shape=(nsamples, horizon, self.action_dim)) * jnp.sqrt(var) + mean
        samples = jnp.clip(samples, self.env.action_space().low, self.env.action_space().high)
        return samples

    @partial(jax.jit, static_argnums=(0,))
    def _generate_num_samples_array(self):
        iterations = jnp.arange(self.agent_config.iCEM_ITERS)
        exp_decay = self.agent_config.BASE_NSAMPS * (self.agent_config.GAMMA ** -iterations)
        min_samples = 2 * self.agent_config.N_ELITES
        samples = jnp.maximum(exp_decay, min_samples)
        samples = jnp.floor(samples).astype(jnp.int32)

        return samples.reshape(-1, 1)

    @partial(jax.jit, static_argnums=(0, 1))
    def _action_space_multi_sample(self, actions_per_plan, key):
        keys = jrandom.split(key, actions_per_plan * self.n_keep)
        keys = keys.reshape((actions_per_plan, self.n_keep))
        sample_fn = lambda k: self.env.action_space().sample(rng=k)
        batched_sample = jax.vmap(sample_fn)
        actions = jax.vmap(batched_sample)(keys)
        return actions

    @partial(jax.jit, static_argnums=(0,))
    def _compute_returns(self, rewards):  # MUST BE SHAPE batch, horizon as polyval uses shape horizon, batch
        # TODO compare against old np.polynomial.polynomial.polyval(discount_factor, rewards)
        return jnp.polyval(rewards.T, self.agent_config.DISCOUNT_FACTOR)  # TODO check discount factor does not change

    @partial(jax.jit, static_argnums=(0, 1, 5, 6))
    def run_algorithm_on_f(self, f, start_obs_O, train_state, key, horizon, actions_per_plan):

        def _outer_loop(outer_loop_state, unused):
            init_obs_O, init_mean_SA, init_var_S1, init_shift_actions_BSA, key = outer_loop_state

            def _iter_iCEM(iCEM_iter_state, unused):
                mean_SA, var_SA, key = iCEM_iter_state

                # loops over the below and then takes trajectories to resample ICEM if not initial
                key, _key = jrandom.split(key)
                init_candidate_actions_BSA = self._iCEM_generate_samples(_key,
                                                                         self.agent_config.NUM_CANDIDATES,
                                                                         self.agent_config.PLANNING_HORIZON,
                                                                         mean_SA,
                                                                         var_SA)

                def _run_single_sample_planning_horizon(init_samples_S1, key):
                    def _run_single_timestep(runner_state, actions_A):
                        obs_O, key = runner_state
                        obsacts_OPA = jnp.concatenate((obs_O, actions_A))
                        key, _key = jrandom.split(key)
                        data_y_O = f(jnp.expand_dims(obsacts_OPA, axis=0), self.env, self.env_params, train_state, _key)
                        nobs_O = self._update_fn(obsacts_OPA, data_y_O, self.env, self.env_params)
                        reward = self.env.reward_function(obsacts_OPA, nobs_O, self.env_params)
                        return (nobs_O, key), MPCTransitionXY(obs=nobs_O,
                                                              action=actions_A,
                                                              reward=jnp.expand_dims(reward, axis=-1),
                                                              x=obsacts_OPA, y=data_y_O)

                    return jax.lax.scan(_run_single_timestep, (init_obs_O, key), init_samples_S1,
                                        self.agent_config.PLANNING_HORIZON)

                init_actions_BSA = jnp.concatenate((init_candidate_actions_BSA, init_shift_actions_BSA), axis=0)
                key, _key = jrandom.split(key)
                batch_key = jrandom.split(_key, self.agent_config.NUM_CANDIDATES + self.n_keep)
                _, planning_traj_BSX = jax.vmap(_run_single_sample_planning_horizon, in_axes=0)(init_actions_BSA,
                                                                                                batch_key)

                # compute return on the entire training list
                all_returns_B = self._compute_returns(jnp.squeeze(planning_traj_BSX.reward, axis=-1))

                # rank returns and chooses the top N_ELITES as the new mean and var
                elite_idx = jnp.argsort(all_returns_B)[-self.agent_config.N_ELITES:]
                elites_ISA = init_actions_BSA[elite_idx]

                new_mean_SA = jnp.mean(elites_ISA, axis=0)
                new_var_SA = jnp.var(elites_ISA, axis=0)

                mpc_transition_xy_BSX = MPCTransitionXY(obs=planning_traj_BSX.obs[elite_idx],
                                                        action=planning_traj_BSX.action[elite_idx],
                                                        reward=planning_traj_BSX.reward[elite_idx],
                                                        x=planning_traj_BSX.x[elite_idx],
                                                        y=planning_traj_BSX.y[elite_idx])

                return (new_mean_SA, new_var_SA, key), mpc_transition_xy_BSX

            (best_mean_SA, best_var_SA, key), iCEM_traj_RISX = jax.lax.scan(_iter_iCEM,
                                                                            (init_mean_SA, init_var_S1, key),
                                                                            None,
                                                                             self.agent_config.iCEM_ITERS)

            iCEM_traj_minus_xy_RISX = MPCTransition(obs=iCEM_traj_RISX.obs,
                                               action=iCEM_traj_RISX.action,
                                               reward=iCEM_traj_RISX.reward)
            iCEM_traj_minus_xy_BSX = jax.tree_util.tree_map(lambda x: jnp.reshape(x,
                                                                                  (x.shape[0] * x.shape[1],
                                                                                   x.shape[2], x.shape[3])),
                                                            iCEM_traj_minus_xy_RISX)

            # find the best sample from iCEM
            all_returns_B = self._compute_returns(jnp.squeeze(iCEM_traj_minus_xy_BSX.reward, axis=-1))
            best_sample_idx = jnp.argmax(all_returns_B)
            best_iCEM_traj_SX = jax.tree_util.tree_map(lambda x: x[best_sample_idx], iCEM_traj_minus_xy_BSX)
            # TODO unsure if this is necessary as the below could also work fine
            # best_iCEM_traj_SX = jax.tree_util.tree_map(lambda x: x[-1, 0], iCEM_traj_RISX)

            # take the number of actions of that plan and add to the existing plan
            planned_iCEM_traj_LX = jax.tree_util.tree_map(lambda x: x[:actions_per_plan], best_iCEM_traj_SX)

            # shift obs
            curr_obs_O = best_iCEM_traj_SX.obs[actions_per_plan-1]

            # shift actions
            keep_indices = jnp.argsort(all_returns_B)[-self.n_keep:]
            short_shifted_actions_BSMLA = iCEM_traj_minus_xy_BSX.action[keep_indices, actions_per_plan:, :]

            # sample new actions and concat onto the "taken" actions
            key, _key = jrandom.split(key)
            new_actions_batch_LBA = self._action_space_multi_sample(actions_per_plan, _key)
            shifted_actions_BSA = jnp.concatenate((short_shifted_actions_BSMLA,
                                                   jnp.swapaxes(new_actions_batch_LBA, 0, 1)), axis=1)

            # remake the mean for iCEM
            end_mean_SA = jnp.concatenate((best_mean_SA[actions_per_plan:], jnp.zeros((actions_per_plan, self.action_dim))))
            end_var_SA = (jnp.ones_like(end_mean_SA) * ((self.env.action_space().high - self.env.action_space().low)
                                                        / self.agent_config.INIT_VAR_DIVISOR) ** 2)

            return (curr_obs_O, end_mean_SA, end_var_SA, shifted_actions_BSA, key), MPCTransitionXYR(obs=planned_iCEM_traj_LX.obs,
                                                                                            action=planned_iCEM_traj_LX.action,
                                                                                            reward=planned_iCEM_traj_LX.reward,
                                                                                            x=iCEM_traj_RISX.x,
                                                                                            y=iCEM_traj_RISX.y,
                                                                                            returns=all_returns_B)

        outer_loop_steps = horizon // actions_per_plan

        init_mean_S1 = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.env.action_space().shape[0]))
        init_var_S1 = (jnp.ones_like(init_mean_S1) * ((self.env.action_space().high - self.env.action_space().low) / self.agent_config.INIT_VAR_DIVISOR) ** 2)
        shift_actions_BSA = jnp.zeros((self.n_keep, self.agent_config.PLANNING_HORIZON, self.action_dim))  # is this okay to add zeros?

        (_, _, _, _, key), overall_traj = jax.lax.scan(_outer_loop, (start_obs_O, init_mean_S1, init_var_S1, shift_actions_BSA, key), None, outer_loop_steps)

        overall_traj_minus_xyr_BLX = MPCTransition(obs=overall_traj.obs, action=overall_traj.action, reward=overall_traj.reward)
        flattened_overall_traj_SX = jax.tree_util.tree_map(lambda x: x.reshape(x.shape[0] * x.shape[1], -1), overall_traj_minus_xyr_BLX)
        # TODO check this flattens correctly aka the batch of L steps merges into a contiguous S

        flatenned_path_x = overall_traj.x.reshape((-1, overall_traj.x.shape[-1]))
        flatenned_path_y = overall_traj.y.reshape((-1, overall_traj.y.shape[-1]))
        # TODO check this actually flattens, do we even want to fllaten this, unsure what shape even is

        joiner_SP1O = jnp.concatenate((jnp.expand_dims(start_obs_O, axis=0), flattened_overall_traj_SX.obs))
        return ((flatenned_path_x, flatenned_path_y),
                (joiner_SP1O, flattened_overall_traj_SX.action, flattened_overall_traj_SX.reward),
                overall_traj.returns)

    @partial(jax.jit, static_argnums=(0,))
    def get_exe_path_crop(self, planned_states, planned_actions):
        obs = planned_states[..., :-1, :]
        nobs = planned_states[..., 1:, :]
        x = jnp.concatenate((obs, planned_actions), axis=-1)
        y = nobs - obs  # TODO this may depend on what the algoirthm outputs, is it diff or is it nobs? Does it?

        return {"exe_path_x": x, "exe_path_y": y}

    @partial(jax.jit, static_argnums=(0, 1, 5, 6))
    def execute_mpc(self, f, obs, train_state, key, horizon, actions_per_plan):
        full_path, output, sample_returns = self.run_algorithm_on_f(f, obs, train_state, key, horizon, actions_per_plan)

        action = output[1]

        exe_path = self.get_exe_path_crop(output[0], output[1])

        return action, exe_path, output

    def make_postmean_func(self):
        def _postmean_fn(x, unused1, unused2, train_state, key):
            mu, std = self.dynamics_model.get_post_mu_cov(x, train_state, full_cov=False)
            return jnp.squeeze(mu, axis=0)
        return _postmean_fn

    def make_postmean_func2(self):
        def _postmean_fn(x, unused1, unused2, train_state, key):
            mu, std = self.dynamics_model.get_post_mu_cov(x, train_state, full_cov=False)
            return mu
        return _postmean_fn

    # @partial(jax.jit, static_argnums=(0,))
    def get_next_point(self, curr_obs_O, train_state, key):
        key, _key = jrandom.split(key)
        action_1A, exe_path, _ = self.execute_mpc(self.make_postmean_func(), curr_obs_O, train_state, _key, horizon=1, actions_per_plan=1)
        x_next_OPA = jnp.concatenate((curr_obs_O, jnp.squeeze(action_1A, axis=0)), axis=-1)

        exe_path = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), exe_path)

        assert jnp.allclose(curr_obs_O, x_next_OPA[:self.obs_dim]), "For rollout cases, we can only give queries which are from the current state"
        # TODO can we jax the assertion?

        return x_next_OPA, exe_path, curr_obs_O, train_state, None, key

def test_MPC_algorithm():
    from project_name.envs.pilco_cartpole import CartPoleSwingUpEnv, pilco_cartpole_reward
    from project_name.envs.gymnax_pilco_cartpole import GymnaxPilcoCartPole
    from project_name.envs.gymnax_pendulum import GymnaxPendulum
    from project_name.envs.wrappers import NormalisedEnv, GenerativeEnv, make_normalised_plot_fn
    from project_name import utils

    # env = CartPoleSwingUpEnv()
    # plan_env = ResettableEnv(CartPoleSwingUpEnv())

    key = jrandom.key(42)

    # env, env_params = gymnax.make("MountainCarContinuous-v0")

    env = GymnaxPilcoCartPole()
    # env = GymnaxPendulum()
    env_params = env.default_params
    env = NormalisedEnv(env, env_params)
    env = GenerativeEnv(env, env_params)

    key, _key = jrandom.split(key)
    start_obs, env_state = env.reset(_key)

    key, _key = jrandom.split(key)
    mpc = MPCAgent(env, env_params, get_config(), key)

    f = utils.get_f_mpc_teleport

    import time
    start_time = time.time()

    path, (observations, actions, rewards), _ = mpc.run_algorithm_on_f(f, start_obs, None, _key,
                                                                        horizon=25,
                                                                        actions_per_plan=mpc.agent_config.ACTIONS_PER_PLAN)
    # batch_key = jrandom.split(_key, 25)
    # path, (observations, actions, rewards) = jax.vmap(mpc.run_algorithm_on_f, in_axes=(None, None, 0))(None, start_obs, batch_key)
    # path, observations, actions, rewards = path[0], observations[0], actions[0], rewards[0]

    print(time.time() - start_time)

    total_return = jnp.sum(rewards)
    print(f"MPC gets {total_return} return with {len(path[0])} queries based on itself")
    done = False
    rewards = []
    for i, action in enumerate(actions):
        next_obs, env_state, rew, done, info = env.step(key, env_state, action, env_params)
        if (next_obs != observations[i+1]).any():
            error = jnp.linalg.norm(next_obs - observations[i+1])
            print(f"i={i}, error={error}")
        rewards.append(rew)
        if done:
            break
    real_return = mpc._compute_returns(jnp.array(rewards))
    print(f"based on the env it gets {real_return} return")

    print(time.time() - start_time)


if __name__ == "__main__":
    test_MPC_algorithm()