"""
Model predictive control (MPC) with BAX.
"""

from argparse import Namespace
import numpy as np
from math import ceil
import logging


from project_name.alg.algorithms import BatchAlgorithm
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
from project_name.utils import MPCTransition, MPCTransitionXY
import gymnax
from project_name.config import get_config


@struct.dataclass  # TODO dodgy for now and need to change
class EnvState(environment.EnvState):
    position: jnp.ndarray
    velocity: jnp.ndarray
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

    @partial(jax.jit, static_argnums=(0,))
    def _iCEM_generate_samples(self, key, nsamples, beta, mean, var, action_lower_bound, action_upper_bound):
        # samples = (colorednoise.powerlaw_psd_gaussian(beta, size=(nsamps, action_dim, horizon)).transpose([0, 2, 1])
        #            * np.sqrt(var) + mean)
        # TODO in future implement this powerlaw_psd_gaussian thing as it seems essential
        samples = jrandom.normal(key, shape=(self.agent_config.BASE_NSAMPS, self.agent_config.PLANNING_HORIZON, self.action_dim)) * jnp.sqrt(var) + mean
        # TODO it is sqrt right?
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

    @partial(jax.jit, static_argnums=(0,))
    def _action_space_multi_sample(self, key, shape):
        # keys = jrandom.split(key, shape[0] * shape[1])
        keys = jrandom.split(key, self.agent_config.ACTIONS_PER_PLAN * self.n_keep)
        keys = keys.reshape((self.agent_config.ACTIONS_PER_PLAN, self.n_keep, -1))
        sample_fn = lambda k: self.env.action_space().sample(rng=k)
        batched_sample = jax.vmap(sample_fn)
        actions = jax.vmap(batched_sample)(keys)
        return actions

    @partial(jax.jit, static_argnums=(0,))
    def _compute_returns(self, rewards):
        # TODO compare against old np.polynomial.polynomial.polyval(discount_factor, rewards)
        return jnp.polyval(rewards, self.agent_config.DISCOUNT_FACTOR)  # TODO check discount factor does not change

    @partial(jax.jit, static_argnums=(0,))
    def run_algorithm_on_f(self, f, init_obs_O, key):
        # TODO is the init_obs okay?

        def _get_f_mpc(x_OPA, use_info_delta=False):  # TODO this should be generalised out of class at some point
            obs_O = x_OPA[:self.obs_dim]
            action_A = x_OPA[self.obs_dim:]
            env_state = EnvState(position=obs_O[0], velocity=obs_O[1],
                                 time=0)  # TODO specific for mountaincar, need to generalise this
            nobs_O, _, _, _, info = self.env.step(key, env_state, action_A, self.env_params)
            # if use_info_delta:  # TODO do we need the info delta?
            #     return info["delta_obs"]
            # else:
            return nobs_O - obs_O

        def _outer_loop(outer_loop_state, unused):
            # this is the one until iter_nums == finished, idk how to calculate this tho
            # it would be some combo of rollout in env dividied by actions_per_plan maybe?

            init_obs_O, init_mean, init_var, init_shift_actions_SB1, key = outer_loop_state

            init_traj = MPCTransition(obs=jnp.zeros((self.agent_config.PLANNING_HORIZON, 1, self.obs_dim)),
                                      action=jnp.zeros((self.agent_config.PLANNING_HORIZON, 1, self.action_dim)),
                                      reward=jnp.zeros((self.agent_config.PLANNING_HORIZON, 1, 1)))  # TODO this may need to be something other than zeros in case of negative rewards
            init_obs_BO = jnp.tile(init_obs_O,
                                   (self.agent_config.BASE_NSAMPS + self.n_keep, 1))  # TODO assumed const sample n

            def _iter_iCEM(iCEM_iter_state, num_samples):
                mean, var, init_saved_traj, key = iCEM_iter_state
                # loops over the below and then takes trajectories to resample ICEM if not initial
                key, _key = jrandom.split(key)  # TODO do I need this?
                init_traj_samples_BS1 = self._iCEM_generate_samples(_key,
                                                           self.agent_config.BASE_NSAMPS,  # TODO have removed adaptive num_samples
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
                    data_y_BO = jax.vmap(_get_f_mpc)(obsacts_BOPA, batch_key)
                    nobs_BO = self._obs_update_fn(obsacts_BOPA, data_y_BO)
                    reward_S1 = jnp.ones((nobs_BO.shape[0], 1))  # self.env.reward_function(new_x_BOPA, nobs_BO) # TODO add the proper reward functions
                    return (nobs_BO, key), MPCTransitionXY(obs=nobs_BO, action=actions_BA, reward=reward_S1,
                                                         x=obsacts_BOPA, y=data_y_BO)
                    # TODO above uses nobs so we never keep track of the initial obs, think this is okay

                init_traj_samples_SB1 = jnp.swapaxes(init_traj_samples_BS1, 0, 1)
                init_traj_samples_SB1 = jnp.concatenate((init_traj_samples_SB1, init_shift_actions_SB1), axis=1)
                (end_obs, key), planning_traj = jax.lax.scan(jax.jit(_run_planning_horizon), (init_obs_BO, key), init_traj_samples_SB1, self.agent_config.PLANNING_HORIZON)

                # need to concat this trajectory with previous ones in the loop ie best_traj
                planning_traj_minus_xy = MPCTransition(obs=planning_traj.obs, action=planning_traj.action, reward=planning_traj.reward)
                traj_combined = jax.tree_util.tree_map(lambda x, y: jnp.concatenate((x, y), axis=1),
                                                       planning_traj_minus_xy,
                                                       init_saved_traj)

                # compute return on the entire training list
                # TODO compare against old np.polynomial.polynomial.polyval(discount_factor, rewards)
                all_returns_BP1 = self._compute_returns(jnp.squeeze(traj_combined.reward, axis=-1))

                # rank returns and chooses the top N_ELITES as the new mean and var
                elite_idx = jnp.argsort(all_returns_BP1)[-self.agent_config.N_ELITES:]  # TODO is this vmappable?
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

            planned_iCEM_traj_flipped = jax.tree_util.tree_map(lambda x: x[:self.agent_config.ACTIONS_PER_PLAN], best_iCEM_traj_flipped)

            curr_obs_O = best_iCEM_traj_flipped.obs[self.agent_config.ACTIONS_PER_PLAN - 1, :]

            # shift_samples
            keep_indices = jnp.argsort(jnp.squeeze(all_returns_R1, axis=-1))[-self.n_keep:]
            short_shifted_actions_S1A = iCEM_traj_flipped.action[self.agent_config.ACTIONS_PER_PLAN:, keep_indices, :]

            key, _key = jrandom.split(key)
            new_actions_batch = self._action_space_multi_sample(_key, (self.agent_config.ACTIONS_PER_PLAN, self.n_keep))

            shifted_actions_S1A = jnp.concatenate((short_shifted_actions_S1A, new_actions_batch))

            end_mean = jnp.concatenate((init_mean[self.agent_config.ACTIONS_PER_PLAN:],
                                        jnp.zeros((self.agent_config.ACTIONS_PER_PLAN, self.action_dim))))
            end_var = (jnp.ones_like(end_mean) * ((self.env.action_space().high - self.env.action_space().low) / self.agent_config.INIT_VAR_DIVISOR) ** 2)

            return (curr_obs_O, end_mean, end_var, shifted_actions_S1A, key), MPCTransitionXY(obs=planned_iCEM_traj_flipped.obs,
                                                                                            action=planned_iCEM_traj_flipped.action,
                                                                                            reward=planned_iCEM_traj_flipped.reward,
                                                                                            x=iCEM_traj.x,
                                                                                            y=iCEM_traj.y)

        outer_loop_steps = self.config.ENV_HORIZON // self.agent_config.ACTIONS_PER_PLAN  # TODO ensure this is an equal division

        init_mean = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.env.action_space().shape[0]))
        init_var = (jnp.ones_like(init_mean) * ((self.env.action_space().high - self.env.action_space().low) / self.agent_config.INIT_VAR_DIVISOR) ** 2)
        shift_actions = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.n_keep, self.action_dim))  # TODO is this okay to add zeros?
        # num_samples = self._generate_num_samples_array()

        (_, _, _, _, key), overall_traj = jax.lax.scan(_outer_loop, (init_obs_O, init_mean, init_var, shift_actions, key), None, outer_loop_steps)

        overall_traj_minus_xy = MPCTransition(obs=overall_traj.obs, action=overall_traj.action, reward=overall_traj.reward)
        flattened_overall_traj = jax.tree_util.tree_map(lambda x: x.reshape(x.shape[0] * x.shape[1], -1), overall_traj_minus_xy)
        # TODO check this flattens correctly

        flatenned_path_x = overall_traj.x.reshape((-1, overall_traj.x.shape[-1]))
        flatenned_path_y = overall_traj.y.reshape((-1, overall_traj.y.shape[-1]))
        # TODO check this actually flattens

        joiner = jnp.concatenate((jnp.expand_dims(init_obs_O, axis=0), flattened_overall_traj.obs))
        return ((flatenned_path_x, flatenned_path_y),
                (joiner, flattened_overall_traj.action, flattened_overall_traj.reward))
        # return ((flatenned_path_x, flatenned_path_y),
        #         (flattened_overall_traj.obs, flattened_overall_traj.action, flattened_overall_traj.reward))

def test_MPC_algorithm():
    from project_name.envs.pilco_cartpole import CartPoleSwingUpEnv, pilco_cartpole_reward
    from project_name.util.control_util import ResettableEnv, get_f_mpc

    # env = CartPoleSwingUpEnv()
    # plan_env = ResettableEnv(CartPoleSwingUpEnv())

    key = jrandom.PRNGKey(42)
    key, _key = jrandom.split(key)
    env, env_params = gymnax.make("MountainCarContinuous-v0")
    start_obs, env_state = env.reset(_key)

    mpc = MPCAgent(env, env_params, get_config(), None, key)
    key, _key = jrandom.split(key)

    import time
    start_time = time.time()

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    path, (observations, actions, rewards) = mpc.run_algorithm_on_f(None, start_obs, _key)
    # batch_key = jrandom.split(_key, 25)
    # path, (observations, actions, rewards) = jax.vmap(mpc.run_algorithm_on_f, in_axes=(None, None, 0))(None, start_obs, batch_key)
    # path, observations, actions, rewards = path[0], observations[0], actions[0], rewards[0]

    print(time.time() - start_time)

    total_return = sum(rewards)
    print(f"MPC gets {total_return} return with {len(path[0])} queries based on itself")
    done = False
    rewards = []
    for i, action in enumerate(actions):
        next_obs, env_state, rew, done, info = env.step(key, env_state, action, env_params)
        if (next_obs != observations[i]).any():
            error = jnp.linalg.norm(next_obs - observations[i])
            print(f"i={i}, error={error}")
        rewards.append(rew)
        if done:
            break
    real_return = compute_return(rewards, 1.0)
    print(f"based on the env it gets {real_return} return")

    print(time.time() - start_time)


if __name__ == "__main__":
    test_MPC_algorithm()