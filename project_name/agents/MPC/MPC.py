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

    def _initialise(self, key):
        # set up initial CEM distribution
        mean = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.env.action_space().shape[0]))
        var = (jnp.ones_like(mean) * ((self.env.action_space().high - self.env.action_space().low)
                                                / self.agent_config.INIT_VAR_DIVISOR) ** 2)
        nsamps = int(max(self.agent_config.BASE_NSAMPS * (self.agent_config.GAMMA ** -1), 2 * self.agent_config.N_ELITES))
        traj_samples = self._iCEM_generate_samples(nsamps,
                                                  self.agent_config.PLANNING_HORIZON,
                                                  self.agent_config.BETA,
                                                  mean,
                                                  var,
                                                  self.env.action_space().low,
                                                  self.env.action_space().high)

        key, _key = jrandom.split(key)
        curr_obs, _ = self.env.reset(_key)

        return curr_obs, traj_samples

    def _obs_update_fn(self, x, y, obs_dim):  # TODO depends on the env, need to instate this somehow, curr no teleport
        start_obs = x[..., :obs_dim]
        delta_obs = y[..., -obs_dim:]
        output = start_obs + delta_obs
        return output  # TODO add teleport stuff and move this into utils probs

    def _first_process_prev_output(self, traj_samples, path_x, path_y, curr_obs_O):
        n_samps = traj_samples.shape[0]
        new_x_SOPA = path_x[-n_samps:]
        new_y_SO = path_y[-n_samps:]
        obs_SO = jnp.tile(curr_obs_O, (n_samps, 1))

        # delta = new_y if self.params.reward_function else new_y[:, 1:]
        delta_SO = new_y_SO

        new_obs_SO = self._obs_update_fn(obs_SO, delta_SO, len(self.env.observation_space(self.env_params).low))
        rewards_S = jnp.ones((new_obs_SO.shape[0]))  # self.env.reward_function(new_x_SOPA, new_obs_SO)
        # TODO add the proper reward functions

        return new_obs_SO, rewards_S

    def _get_sample_x_batch(self, traj_samples, current_t_plan):
        actions = traj_samples[:, current_t_plan, :]
        obs = self.traj_states[-1]
        queries = np.concatenate([obs, actions], axis=1)
        batch = list(queries)

        return batch

    def _get_next_x_batch(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        if len(self.exe_path.x) > 0:
            self.process_prev_output()
        # at this point the *_done should be correct
        if self.samples_done and self.iter_num + 1 == self.params.num_iters:
            shift_actions = self.save_planned_actions()
            if self.current_t >= self.params.env_horizon:
                # done planning
                return []
            self.reset_CEM(shift_actions)
        elif self.samples_done:
            self.resample_iCEM()
        return self.get_sample_x_batch()

    def run_algorithm_on_f(self, f, key):
        # initialise the MPC
        curr_obs_O, init_traj_samples = self._initialise(key)

        # actions_BA = init_traj_samples[:, 0, :]
        # obs_BO = jnp.tile(curr_obs_O, (actions_SA.shape[0], 1))
        # batch_x_SOPA = jnp.concatenate([obs_SO, actions_SA], axis=1)
        #
        # key, _key = jrandom.split(key)
        # batched_key = jrandom.split(_key, batch_x_SOPA.shape[0])
        # batch_y_SO = f(batch_x_SOPA, batched_key)
        #
        # new_obs_SO, rewards_S = self._first_process_prev_output(init_traj_samples, batch_x_SOPA, batch_y_SO, curr_obs_O)
        #
        # batch = self._get_sample_x_batch(init_traj_samples, 0)
        #
        # self._get_next_x_batch()

        """
        This will be a scan
        
        Initially init CEM trajectories and instantiate some first step obs
        Shape is batch, planning_horizon, ...
        
        then loop begins:
        use initial obs
        add the actions from the cem trajectory for that step
        run over the dynamics model to get y
        update_fn to get nobs aka obs + y = nobs
        get reward
        end loop
        
        resample_iCEM and run loop again, CAN'T BATCH AS IT REQUIRES PREVIOUS LOOP DETAILS
        
        somehow figure out another outer loop on top of it but I think that leads off each other        
        """

        def outer_loop():
            # this is the one until iter_nums == finished, idk how to calculate this tho
            return

        def sample_ICEM():
            # loops over the below and then takes trajectories to resample ICEM if not initial
            return

        def _run_planning_horizon(runner_state, unused):
            obs, actions = runner_state
            obsacts = jnp.concatenate((obs, actions))
            data_y = f(obsacts)
            nobs = update_fn(obsacts, data_y)
            return (nobs), Transition()

        ((train_state, mem_state, env_state, last_obs_NO, last_done_N, key),
         trajectory_batch_LNZ) = jax.lax.scan(_run_planning_horizon, runner_state, None,
                                              actor.agent.agent_config.NUM_INNER_STEPS)

        jax.vmap(run_planning_horizon)

        return


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