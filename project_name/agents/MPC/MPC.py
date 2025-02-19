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
from project_name.utils import MPCTransition


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
        # TODO match this to the other rl main stuff

        # # self.traj_samples = list(self.traj_samples)
        # # self.traj_samples += samples_to_pass
        # # this one is for CEM
        # self.current_t_plan = 0
        # # this one is for the actual agent
        # self.current_t = 0
        # if self.params.start_obs is not None:
        #     logging.debug("Copying given start obs")
        #     self.current_obs = self.params.start_obs
        # else:
        #     logging.debug("Sampling start obs from env")
        #     self.current_obs = self.params.env.reset()
        # self.iter_num = 0
        # self.samples_done = False
        # self.planned_states = [self.current_obs]
        # self.planned_actions = []
        # self.planned_rewards = []
        # self.saved_states = []
        # self.saved_actions = []
        # self.saved_rewards = []
        # self.traj_states = []
        # self.traj_rewards = []
        # self.best_return = -np.inf
        # self.best_actions = None
        # self.best_obs = None
        # self.best_rewards = None

    def create_train_state(self):  # TODO what would this be in the end?
        return None, None

    # @partial(jax.jit, static_argnums=(0,))
    def _iCEM_generate_samples(self, nsamps, horizon, beta, mean, var, action_lower_bound, action_upper_bound):
        action_dim = mean.shape[-1]
        samples = (colorednoise.powerlaw_psd_gaussian(beta, size=(nsamps, action_dim, horizon)).transpose([0, 2, 1])
                   * np.sqrt(var) + mean)
        samples = np.clip(samples, action_lower_bound, action_upper_bound)
        return samples

    def _initialise(self, iter_num, mean, var, key):
        # set up initial CEM distribution
        nsamps = int(max(self.agent_config.BASE_NSAMPS * (self.agent_config.GAMMA ** -iter_num), 2 * self.agent_config.N_ELITES))
        traj_samples = self._iCEM_generate_samples(nsamps,
                                                  self.agent_config.PLANNING_HORIZON,
                                                  self.agent_config.BETA,
                                                  mean,
                                                  var,
                                                  self.env.action_space().low,
                                                  self.env.action_space().high)

        key, _key = jrandom.split(key)
        curr_obs, _ = self.env.reset(_key)

        return curr_obs, traj_samples, key

    def get_f_mpc(self, use_info_delta=False):  # TODO this should be generalised out of class at some point
        def f(x_OPA, key):
            obs_O = x_OPA[:self.obs_dim]
            action_A = x_OPA[self.obs_dim:]
            env_state = EnvState(position=obs_O[0], velocity=obs_O[1], time=0)  # TODO specific for mountaincar, need to generalise this
            nobs_O, _, _, _, info = self.env.step(key, env_state, action_A, self.env_params)
            if use_info_delta:
                return info["delta_obs"]
            else:
                return nobs_O - obs_O

        return jax.vmap(f)  # TODO we have to vmap this due to the iCEM samples, can we do this last?

    def _obs_update_fn(self, x, y):  # TODO depends on the env, need to instate this somehow, curr no teleport
        start_obs = x[..., :self.obs_dim]
        delta_obs = y[..., -self.obs_dim:]
        output = start_obs + delta_obs
        return output  # TODO add teleport stuff and move this into utils probs

    def run_algorithm_on_f(self, f, key):
        f = self.get_f_mpc()

        def outer_loop():
            # this is the one until iter_nums == finished, idk how to calculate this tho
            return

        def _iter_iCEM(iCEM_iter_state, unused):
            iter_nums, mean, var, saved_traj, key = iCEM_iter_state
            # loops over the below and then takes trajectories to resample ICEM if not initial
            init_obs_O, init_traj_samples_BS1, key = self._initialise(iter_nums, mean, var, key)  # TODO does key have to run throughout?

            def _run_planning_horizon(runner_state, actions_BA):
                obs_BO, key = runner_state
                key, _key = jrandom.split(key)
                obsacts_BOPA = jnp.concatenate((obs_BO, actions_BA), axis=-1)
                batch_key = jrandom.split(_key, obs_BO.shape[0])
                data_y_BO = f(obsacts_BOPA, batch_key)
                nobs_BO = self._obs_update_fn(obsacts_BOPA, data_y_BO)
                reward_S1 = jnp.ones((nobs_BO.shape[0], 1))  # self.env.reward_function(new_x_BOPA, nobs_BO) # TODO add the proper reward functions
                return (nobs_BO, key), MPCTransition(obs=obs_BO, action=actions_BA, reward=reward_S1)

            init_traj_samples_SB1 = jnp.swapaxes(init_traj_samples_BS1, 0, 1)
            init_obs_BO = jnp.tile(init_obs_O, (init_traj_samples_BS1.shape[0], 1))
            (end_obs, key), planning_traj = jax.lax.scan(_run_planning_horizon, (init_obs_BO, key), init_traj_samples_SB1, self.agent_config.PLANNING_HORIZON)

            # need to concat this trajectory with previous ones in the loop ie best_traj
            all_rewards = jnp.concatenate(planning_traj.reward, saved_traj.reward)
            all_states = jnp.concatenate(planning_traj.obs, saved_traj.obs)  # TODO it may actually only want the nobs
            all_actions = jnp.concatenate(planning_traj.action, saved_traj.action)

            # compute return on the entire training list
            all_returns = compute_return(all_rewards, self.agent_config.DISCOUNT_FACTOR)

            # best_current_return =
            # best_action = etc etc
            # TODO i think we can just do the best at the end after all iterations maybe?

            # rank returns and find out the elite_idx and the correct elites
            elite_idx = jnp.argsort(all_returns)[-self.agent_config.N_ELITES]  # TODO is this vmappable?
            elites = all_actions[elite_idx, ...]

            mean = np.mean(elites, axis=0)
            var = np.var(elites, axis=0)

            iter_nums += 1

            n_save_elites = ceil(self.agent_config.N_ELITES * self.agent_config.XI)
            save_idx = elite_idx[-n_save_elites:]
            saved_rewards = all_actions[save_idx, ...]
            saved_states = all_states[save_idx, ...]
            saved_actions = all_actions[save_idx, ...]

            return (iter_nums, mean, var, key), MPCTransition()

        init_mean = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.env.action_space().shape[0]))
        init_var = (jnp.ones_like(init_mean) * ((self.env.action_space().high - self.env.action_space().low) / self.agent_config.INIT_VAR_DIVISOR) ** 2)
        (_, _, _, key), iCEM_traj = jax.lax.scan(_iter_iCEM, (0, init_mean, init_var, key), None, self.agent_config.iCEM_ITERS)



        return

    def resample_iCEM(self):
        self.iter_num += 1
        nsamps = int(
            max(
                self.params.base_nsamps * (self.params.gamma ** -self.iter_num),
                2 * self.params.n_elites,
            )
        )
        if len(self.saved_rewards) > 0:
            all_rewards = np.concatenate(
                [np.array(self.traj_rewards).T, np.array(self.saved_rewards)], axis=0
            )
            all_states = np.concatenate(
                [
                    np.array(self.traj_states).transpose((1, 0, 2)),
                    np.array(self.saved_states),
                ],
                axis=0,
            )
            all_actions = np.concatenate(
                [self.traj_samples, self.saved_actions], axis=0
            )
        else:
            all_rewards = np.array(self.traj_rewards).T
            all_states = np.array(self.traj_states).transpose((1, 0, 2))
            all_actions = self.traj_samples

        all_returns = compute_return(all_rewards, self.params.discount_factor)
        best_idx = np.argmax(all_returns)
        best_current_return = all_returns[best_idx]
        if best_current_return > self.best_return:
            self.best_return = best_current_return
            self.best_actions = all_actions[best_idx, ...]
            self.best_obs = all_states[best_idx, ...]
            self.best_rewards = all_rewards[best_idx, ...]
        elite_idx = np.argsort(all_returns)[-self.params.n_elites:]
        elites = all_actions[elite_idx, ...]
        mean = np.mean(elites, axis=0)
        var = np.var(elites, axis=0)
        samples = iCEM_generate_samples(
            nsamps,
            self.params.planning_horizon,
            self.params.beta,
            mean,
            var,
            self.params.action_lower_bound,
            self.params.action_upper_bound,
        )
        n_save_elites = ceil(self.params.n_elites * self.params.xi)
        save_idx = elite_idx[-n_save_elites:]
        self.saved_actions = all_actions[save_idx, ...]
        self.saved_states = all_states[save_idx, ...]
        self.saved_rewards = all_rewards[save_idx, ...]
        if self.iter_num + 1 == self.params.num_iters:
            samples = np.concatenate([samples, mean[None, :]], axis=0)
        self.traj_samples = samples
        # self.traj_samples = list(samples)
        self.traj_states = []
        self.traj_rewards = []
        self.samples_done = False


def test_MPC_algorithm():
    from project_name.envs.pilco_cartpole import CartPoleSwingUpEnv, pilco_cartpole_reward
    from project_name.util.control_util import ResettableEnv, get_f_mpc

    env = CartPoleSwingUpEnv()
    plan_env = ResettableEnv(CartPoleSwingUpEnv())
    f = get_f_mpc(plan_env)
    start_obs = env.reset()
    params = dict(start_obs=start_obs, env=plan_env, reward_function=pilco_cartpole_reward)
    mpc = MPC(params)
    mpc.initialize()
    path, output = mpc.run_algorithm_on_f(f)
    observations, actions, rewards = output
    total_return = sum(rewards)
    print(f"MPC gets {total_return} return with {len(path.x)} queries based on itself")
    done = False
    rewards = []
    for i, action in enumerate(actions):
        next_obs, rew, done, info = env.step(action)
        if (next_obs != observations[i + 1]).any():
            error = np.linalg.norm(next_obs - observations[i + 1])
            print(f"i={i}, error={error}")
        rewards.append(rew)
        if done:
            break
    real_return = compute_return(rewards, 1.0)
    print(f"based on the env it gets {real_return} return")


if __name__ == "__main__":
    test_MPC_algorithm()