"""
Model predictive control (MPC) with BAX.
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


@struct.dataclass  # TODO dodgy for now and need to change
class EnvState(environment.EnvState):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int


class MPCAgent(AgentBase):
    """
    An algorithm for model-predictive control. Here, the queries are concatenated states
    and actions and the output of the query is the next state.  We need the reward
    function in our algorithm as well as a start state.
    """

    def __init__(self, env, env_params, config, utils, key):
        self.config = config
        self.agent_config = get_MPC_config()
        self.env = env
        self.env_params = env_params

        self.obs_dim = len(self.env.observation_space(self.env_params).low)
        self.action_dim = self.env.action_space().shape[0]
        # TODO match this to the other rl main stuff

        self.n_keep = ceil(self.agent_config.XI * self.agent_config.N_ELITES)

    def create_train_state(self):  # TODO what would this be in the end?
        return None, None

    @partial(jax.jit, static_argnums=(0, 3, 4, 5))  # TODO should really verify this
    def powerlaw_psd_gaussian_jax(self,
            key: jrandom.PRNGKey,
            exponent: float,
            base_nsamps: int,
            action_dim: int,
            time_horizon: int,
            fmin: float = 0.0,
    ) -> jnp.ndarray:
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
    def _iCEM_generate_samples(self, key, nsamples, horizon, beta, mean, var, action_lower_bound, action_upper_bound):
        # samples = (colorednoise.powerlaw_psd_gaussian(beta, size=(nsamps, action_dim, horizon)).transpose([0, 2, 1])
        #            * np.sqrt(var) + mean)
        samples = jnp.swapaxes(self.powerlaw_psd_gaussian_jax(key, beta, nsamples, self.action_dim, horizon),
                               1, 2)  * jnp.sqrt(var) + mean
        # TODO test this powerlaw thing
        # samples = jrandom.normal(key, shape=(nsamples, horizon, self.action_dim)) * jnp.sqrt(var) + mean
        samples = jnp.clip(samples, action_lower_bound, action_upper_bound)
        return samples

    @partial(jax.jit, static_argnums=(0,))
    def _generate_num_samples_array(self):
        iterations = jnp.arange(self.agent_config.iCEM_ITERS)
        exp_decay = self.agent_config.BASE_NSAMPS * (self.agent_config.GAMMA ** -iterations)
        min_samples = 2 * self.agent_config.N_ELITES
        samples = jnp.maximum(exp_decay, min_samples)
        samples = jnp.floor(samples).astype(jnp.int32)

        return samples.reshape(-1, 1)

    @partial(jax.jit, static_argnums=(0,))
    def _obs_update_fn(self, x, y):  # TODO depends on the env, need to instate this somehow, curr no teleport
        start_obs = x[..., :self.obs_dim]
        delta_obs = y[..., -self.obs_dim:]
        output = start_obs + delta_obs
        return output  # TODO add teleport stuff and move this into utils probs

    @partial(jax.jit, static_argnums=(0, 2))
    def _action_space_multi_sample(self, key, actions_per_plan):
        keys = jrandom.split(key, actions_per_plan * self.n_keep)
        keys = keys.reshape((actions_per_plan, self.n_keep, -1))
        sample_fn = lambda k: self.env.action_space().sample(rng=k)
        batched_sample = jax.vmap(sample_fn)
        actions = jax.vmap(batched_sample)(keys)
        return actions

    @partial(jax.jit, static_argnums=(0,))
    def _compute_returns(self, rewards):
        # TODO compare against old np.polynomial.polynomial.polyval(discount_factor, rewards)
        return jnp.polyval(rewards, self.agent_config.DISCOUNT_FACTOR)  # TODO check discount factor does not change

    @partial(jax.jit, static_argnums=(0, 1, 4, 5))
    def run_algorithm_on_f(self, f, init_obs_O, key, horizon, actions_per_plan):
        # def _get_f_mpc(x_OPA, use_info_delta=False):  # TODO this should be generalised out of class at some point
        #     obs_O = x_OPA[:self.obs_dim]
        #     action_A = x_OPA[self.obs_dim:]
        #     env_state = EnvState(x=obs_O[0], x_dot=obs_O[1], theta=obs_O[2], theta_dot=obs_O[3],
        #                          time=0)  # TODO specific for cartpole, need to generalise this
        #     nobs_O, _, _, _, info = self.env.step(key, env_state, action_A, self.env_params)
        #     # if use_info_delta:  # TODO do we need the info delta?
        #     #     return info["delta_obs"]
        #     # else:
        #     return nobs_O - obs_O

        def _outer_loop(outer_loop_state, unused):
            init_obs_O, init_mean, init_var, init_shift_actions_SB1, key = outer_loop_state

            init_traj = MPCTransition(obs=jnp.zeros((self.agent_config.PLANNING_HORIZON, 1, self.obs_dim)),
                                      action=jnp.zeros((self.agent_config.PLANNING_HORIZON, 1, self.action_dim)),
                                      reward=jnp.ones((self.agent_config.PLANNING_HORIZON, 1, 1)) * -100)   # TODO this may need to be something other than zeros in case of negative rewards
            init_obs_BO = jnp.tile(init_obs_O,(self.agent_config.BASE_NSAMPS + self.n_keep, 1))
            # TODO assumed const sample n for now

            def _iter_iCEM(iCEM_iter_state, num_samples):
                mean, var, init_saved_traj, key = iCEM_iter_state
                # loops over the below and then takes trajectories to resample ICEM if not initial
                key, _key = jrandom.split(key)  # TODO do I need this?
                init_traj_samples_BS1 = self._iCEM_generate_samples(_key,
                                                           self.agent_config.BASE_NSAMPS,  # TODO have removed adaptive num_samples
                                                           self.agent_config.PLANNING_HORIZON,
                                                           self.agent_config.BETA,
                                                           mean,
                                                           var,
                                                           self.env.action_space().low,
                                                           self.env.action_space().high)

                def _run_planning_horizon(runner_state, actions_BA):
                    obs_BO, key = runner_state
                    key, _key = jrandom.split(key)
                    obsacts_BOPA = jnp.concatenate((obs_BO, actions_BA), axis=-1)
                    batch_key = jrandom.split(_key, obs_BO.shape[0])
                    data_y_BO = jax.vmap(f)(obsacts_BOPA, batch_key)  # TODO check what this vmap is for, can we vmap outside the whole loop?
                    # data_y_BO = jax.vmap(_get_f_mpc)(obsacts_BOPA, batch_key)
                    nobs_BO = self._obs_update_fn(obsacts_BOPA, data_y_BO)
                    reward_S1 = self.env.reward_function(obsacts_BOPA, nobs_BO, self.env_params)
                    return (nobs_BO, key), MPCTransitionXY(obs=nobs_BO, action=actions_BA, reward=jnp.expand_dims(reward_S1, axis=-1),
                                                         x=obsacts_BOPA, y=data_y_BO)
                    # TODO maybe remove the extra reward dim if unneccesary

                init_traj_samples_SB1 = jnp.swapaxes(init_traj_samples_BS1, 0, 1)
                init_traj_samples_SB1 = jnp.concatenate((init_traj_samples_SB1, init_shift_actions_SB1), axis=1)
                (end_obs, key), planning_traj = jax.lax.scan(jax.jit(_run_planning_horizon), (init_obs_BO, key),
                                                             init_traj_samples_SB1, self.agent_config.PLANNING_HORIZON)

                # need to concat this trajectory with previous ones in the loop ie best_traj
                planning_traj_minus_xy = MPCTransition(obs=planning_traj.obs, action=planning_traj.action, reward=planning_traj.reward)
                traj_combined = jax.tree_util.tree_map(lambda x, y: jnp.concatenate((x, y), axis=1),
                                                       planning_traj_minus_xy,
                                                       init_saved_traj)

                # compute return on the entire training list
                # TODO compare against old np.polynomial.polynomial.polyval(discount_factor, rewards)
                all_returns_BP1 = self._compute_returns(jnp.squeeze(traj_combined.reward, axis=-1))

                # rank returns and chooses the top N_ELITES as the new mean and var
                elite_idx = jnp.argsort(all_returns_BP1)[-self.agent_config.N_ELITES:]
                elites_SIA = traj_combined.action[:, elite_idx, ...]

                mean_SA = jnp.mean(elites_SIA, axis=1)
                var_SA = jnp.var(elites_SIA, axis=1)

                save_idx = elite_idx[-self.n_keep:]
                end_traj_saved = jax.tree_util.tree_map(lambda x: x[:, save_idx, ...], traj_combined)
                # TODO surely saved contains the best option as well right?

                mpc_transition = MPCTransition(obs=end_traj_saved.obs,
                                               action=end_traj_saved.action,
                                               reward=end_traj_saved.reward)

                mpc_transition_xy = MPCTransitionXY(obs=end_traj_saved.obs,
                                                 action=end_traj_saved.action,
                                                 reward=end_traj_saved.reward,
                                                 x=planning_traj.x,
                                                 y=planning_traj.y)
                # TODO kinda dodge but it works, can we streamline?

                return (mean_SA, var_SA, mpc_transition, key), mpc_transition_xy

            (_, _, _, key), iCEM_traj = jax.lax.scan(_iter_iCEM,(init_mean, init_var, init_traj, key),
                                                     None,
                                                     self.agent_config.iCEM_ITERS)

            iCEM_traj_minus_xy = MPCTransition(obs=iCEM_traj.obs, action=iCEM_traj.action, reward=iCEM_traj.reward)
            iCEM_traj_flipped = jax.tree_util.tree_map(lambda x: jnp.swapaxes(jnp.squeeze(x, axis=-2), 0, 1), iCEM_traj_minus_xy)

            all_returns_R1 = self._compute_returns(iCEM_traj_flipped.reward)
            best_sample_idx = jnp.argmax(all_returns_R1)
            best_iCEM_traj_flipped = jax.tree_util.tree_map(lambda x: x[:, best_sample_idx, ...], iCEM_traj_flipped)

            planned_iCEM_traj_flipped = jax.tree_util.tree_map(lambda x: x[:actions_per_plan], best_iCEM_traj_flipped)

            curr_obs_O = best_iCEM_traj_flipped.obs[actions_per_plan - 1, :]

            # shift_samples
            keep_indices = jnp.argsort(jnp.squeeze(all_returns_R1, axis=-1))[-self.n_keep:]
            short_shifted_actions_S1A = iCEM_traj_flipped.action[actions_per_plan:, keep_indices, :]

            key, _key = jrandom.split(key)
            new_actions_batch = self._action_space_multi_sample(_key, actions_per_plan)

            shifted_actions_S1A = jnp.concatenate((short_shifted_actions_S1A, new_actions_batch))

            end_mean = jnp.concatenate((init_mean[actions_per_plan:],
                                        jnp.zeros((actions_per_plan, self.action_dim))))
            end_var = (jnp.ones_like(end_mean) * ((self.env.action_space().high - self.env.action_space().low) / self.agent_config.INIT_VAR_DIVISOR) ** 2)

            return (curr_obs_O, end_mean, end_var, shifted_actions_S1A, key), MPCTransitionXYR(obs=planned_iCEM_traj_flipped.obs,
                                                                                            action=planned_iCEM_traj_flipped.action,
                                                                                            reward=planned_iCEM_traj_flipped.reward,
                                                                                            x=iCEM_traj.x,
                                                                                            y=iCEM_traj.y,
                                                                                            returns=all_returns_R1)

        outer_loop_steps = horizon // actions_per_plan  # TODO ensure this is an equal division

        init_mean = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.env.action_space().shape[0]))
        init_var = (jnp.ones_like(init_mean) * ((self.env.action_space().high - self.env.action_space().low) / self.agent_config.INIT_VAR_DIVISOR) ** 2)
        shift_actions = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.n_keep, self.action_dim))  # TODO is this okay to add zeros?
        # num_samples = self._generate_num_samples_array()

        (_, _, _, _, key), overall_traj = jax.lax.scan(_outer_loop, (init_obs_O, init_mean, init_var, shift_actions, key), None, outer_loop_steps)

        overall_traj_minus_xyr = MPCTransition(obs=overall_traj.obs, action=overall_traj.action, reward=overall_traj.reward)
        flattened_overall_traj = jax.tree_util.tree_map(lambda x: x.reshape(x.shape[0] * x.shape[1], -1), overall_traj_minus_xyr)
        # TODO check this flattens correctly

        flatenned_path_x = overall_traj.x.reshape((-1, overall_traj.x.shape[-1]))
        flatenned_path_y = overall_traj.y.reshape((-1, overall_traj.y.shape[-1]))
        # TODO check this actually flattens

        joiner = jnp.concatenate((jnp.expand_dims(init_obs_O, axis=0), flattened_overall_traj.obs))
        return ((flatenned_path_x, flatenned_path_y),
                (joiner, flattened_overall_traj.action, flattened_overall_traj.reward),
                overall_traj.returns)

    def get_exe_path_crop(self, planned_states, planned_actions):
        obs = planned_states[:-1]
        nobs = planned_states[1:]
        x = jnp.concatenate((obs, planned_actions), axis=-1)
        y = nobs - obs

        # TODO add some clip if it goes outside the domain and project to domain as well as terminal sorting out

        return {"exe_path_x": x, "exe_path_y": y}

    def execute_mpc(self, f, obs, key):

        # TODO add some if statement and stuff if it will be open-loop

        full_path, output, _ = self.run_algorithm_on_f(f, obs, key, horizon=1, actions_per_plan=1)

        action = output[1]

        exe_path = self.get_exe_path_crop(output[0], output[1])

        return action, exe_path

    def optimise(self, dynamics_model, dynamics_model_params, f, exe_path_BSOPA, x_test, key):
        curr_obs_O = x_test[:self.obs_dim]
        mean = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.action_dim))  # TODO this may not be zero if there is alreayd an action sequence, should check this
        init_var_divisor = 4
        var = jnp.ones_like(mean) * ((self.env.action_space().high - self.env.action_space().low) / init_var_divisor) ** 2

        def _iter_iCEM2(iCEM2_runner_state, unused):  # TODO perhaps we can generalise this from above
            mean_S1, var_S1, prev_samples, prev_returns, key = iCEM2_runner_state
            key, _key = jrandom.split(key)  # TODO do I need this?
            samples_BS1 = self._iCEM_generate_samples(_key,
                                                      self.agent_config.BASE_NSAMPS,
                                                      self.agent_config.PLANNING_HORIZON,
                                                      self.agent_config.BETA,
                                                      mean_S1,
                                                      var_S1,
                                                      self.env.action_space().low,
                                                      self.env.action_space().high)

            key, _key = jrandom.split(key)
            batch_key = jrandom.split(_key, self.agent_config.BASE_NSAMPS)  # TODO do we need this double split?
            acq = jax.vmap(self._evaluate_samples, in_axes=(None, None, None, None, 0, None, 0))(dynamics_model,
                                                                                               dynamics_model_params,
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

    def _evaluate_samples(self, dynamics_model, dynamics_model_params, f, obs_O, samples_S1, exe_path_BSOPA, key):
        # run a for loop planning basically
        def _run_planning_horizon2(runner_state, actions_A):  # TODO again can we generalise this from above to save rewriting things
            obs_O, key = runner_state
            obsacts_OPA = jnp.concatenate((obs_O, actions_A), axis=-1)
            key, _key = jrandom.split(key)
            data_y_O = f(obsacts_OPA, _key)
            nobs_O = self._obs_update_fn(obsacts_OPA, data_y_O)
            return (nobs_O, key), obsacts_OPA

        _, x_list_SOPA = jax.lax.scan(jax.jit(_run_planning_horizon2), (obs_O, key), samples_S1)

        # this part is the acquisition function so should be generalised at some point rather than putting it here
        # TODO add this as a function

        # get posterior covariance for x_set
        _, post_cov = dynamics_model.get_post_mu_cov(x_list_SOPA, dynamics_model_params, full_cov=True)

        # get posterior covariance for all exe_paths, so this be a vmap probably
        def _get_sample_cov(x_list_SOPA, exe_path_SOPA, params):
            train_data = jnp.concatenate((params["train_data"], exe_path_SOPA["exe_path_x"]))
            params["q_mu"] = jnp.zeros((train_data.shape[0], params["q_mu"].shape[-1]))  # TODO a dodgy workaround for now
            return dynamics_model.get_post_mu_cov(x_list_SOPA, params, train_data=train_data, full_cov=True)

        _, samp_cov = jax.vmap(_get_sample_cov, in_axes=(None, 0, None))(x_list_SOPA, exe_path_BSOPA, dynamics_model_params)

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

def test_MPC_algorithm():
    from project_name.envs.pilco_cartpole import CartPoleSwingUpEnv, pilco_cartpole_reward
    from project_name.util.control_util import ResettableEnv, get_f_mpc
    from project_name.envs.gymnax_pilco_cartpole import GymnaxPilcoCartPole

    # env = CartPoleSwingUpEnv()
    # plan_env = ResettableEnv(CartPoleSwingUpEnv())

    key = jrandom.PRNGKey(42)
    key, _key = jrandom.split(key)

    # env, env_params = gymnax.make("MountainCarContinuous-v0")

    env = GymnaxPilcoCartPole()
    env_params = env.default_params
    obs_dim = len(env.observation_space(env_params).low)

    start_obs, env_state = env.reset(_key)

    mpc = MPCAgent(env, env_params, get_config(), None, key)
    key, _key = jrandom.split(key)

    def _get_f_mpc(x_OPA, key, use_info_delta=False):  # TODO this should be generalised out of class at some point
        obs_O = x_OPA[:obs_dim]
        action_A = x_OPA[obs_dim:]
        env_state = EnvState(x=obs_O[0], x_dot=obs_O[1], theta=obs_O[2], theta_dot=obs_O[3], time=0)  # TODO specific for cartpole, need to generalise this
        nobs_O, _, _, _, info = env.step(key, env_state, action_A, env_params)
        return nobs_O - obs_O

    f = _get_f_mpc  # TODO kinda weak but okay for now

    import time
    start_time = time.time()

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    with jax.disable_jit(disable=False):
        path, (observations, actions, rewards), _ = mpc.run_algorithm_on_f(f, start_obs, _key,
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
    real_return = compute_return(rewards, 1.0)
    print(f"based on the env it gets {real_return} return")

    print(time.time() - start_time)


if __name__ == "__main__":
    test_MPC_algorithm()