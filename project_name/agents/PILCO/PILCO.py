"""
Based off the following code "https://github.com/mathDR/jax-pilco/blob/main/pilco/models/pilco.py" and original paper
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
import jax
from functools import partial
# import colorednoise
import jax.random as jrandom
from gymnax.environments import environment
from flax import struct
from project_name.utils import MPCTransition, MPCTransitionXY, MPCTransitionXYR
import gymnax
from project_name.config import get_config
from typing import Union, Tuple
from project_name.utils import update_obs_fn, update_obs_fn_teleport, get_f_mpc, get_f_mpc_teleport
from project_name import dynamics_models
from project_name.agents.MPC import MPCAgent
from project_name.agents.PILCO import LinearController, get_PILCO_config
import optax
from flax.training.train_state import TrainState


class PILCOAgent(MPCAgent):
    """
    Give it some initial training data
    Fit the hyperparams to this initial data for both the dynamics model and controller
    Traditionally then rollout a plan ~25 steps or so
    Then predict the reward maybe?
    Add all this data to the gp
    Then optimise

    New setup
    Get initial data and fit the hyperparams
    Do the usual main_loop for x steps using actor.get_next_point
    When we get to 25 steps than actually optimise after getting the next point
    It imports from MPC for ground truth creation, is that okay idk?
    """
    def __init__(self, env, env_params, config, key):
        super().__init__(env, env_params, config, key)
        self.agent_config = get_PILCO_config()

        # TODO add some import from folder check thingo
        self.dynamics_model = dynamics_models.MOGP(env, env_params, config, self.agent_config, key)
        # self.dynamics_model = dynamics_models.MOGPGPJax(env, env_params, config, self.agent_config, key)
        # self.dynamics_model = dynamics_models.MOSVGP(env, env_params, config, self.agent_config, key)

        self.obs_dim = len(self.env.observation_space(self.env_params).low)
        self.action_dim = self.env.action_space().shape[0]
        # TODO match this to the other rl main stuff

        self.controller = LinearController(self.obs_dim, self.action_dim, self.env.action_space().high)

        self.tx = optax.adam(self.agent_config.LR)

        if config.TELEPORT:
            self._update_fn = update_obs_fn_teleport
        else:
            self._update_fn = update_obs_fn

        self.m_init = jnp.reshape(jnp.array([0.0, 0.0]), (1, 2))
        self.S_init = jnp.diag(jnp.array([0.05, 0.01]))
        # TODO both above are for pendulum, can we generalise

    def create_train_state(self, init_data_x, init_data_y, key):
        key, _key = jrandom.split(key)
        dynamics_train_state = self.dynamics_model.create_train_state(init_data_x, init_data_y, _key)
        # TODO check if need the split above

        key, _key = jrandom.split(key)
        params = self.controller.init(_key,
                                      jnp.zeros((1, self.obs_dim)),
                                      jnp.zeros((self.obs_dim, self.obs_dim)))
        controller_train_state = TrainState.create(apply_fn=self.controller.apply, params=params, tx=self.tx)
        dynamics_train_state["controller_train_state"] = controller_train_state

        return dynamics_train_state

    def pretrain_params(self, init_data_x, init_data_y, pretrain_data_x, pretrain_data_y, key):
        # create train_state
        train_state = self.create_train_state(init_data_x, init_data_y, key)

        # optimisation for the dynamics model
        opt_posterior = self.dynamics_model.pretrain_params(init_data_x, init_data_y, pretrain_data_x, pretrain_data_y, key)
        train_state["posterior"] = opt_posterior

        # controller optimisation
        controller_params = self._optimise_policy(train_state)
        train_state["controller_train_state"] = controller_params

        return train_state

    def _optimise_policy(self, train_state, maxiter=1000, restarts=1):
        def training_loss():
            # This is for tuning controller's parameters
            init_val = (self.m_init, self.S_init, 0.0)

            def body_fun(i, v):
                m_x, s_x, reward = v

                def _propogate(m_x, s_x):
                    m_u, s_u, c_xu = self.controller.apply(train_state["controller_train_state"].params, m_x, s_x)

                    m = jnp.concatenate([m_x, m_u], axis=1)
                    s1 = jnp.concatenate([s_x, s_x @ c_xu], axis=1)
                    s2 = jnp.concatenate([jnp.transpose(s_x @ c_xu), s_u], axis=1)
                    s = jnp.concatenate([s1, s2], axis=0)

                    M_dx, S_dx, C_dx = self.dynamics_model.predict_on_noisy_inputs(m, s, train_state)
                    M_x = M_dx + m_x
                    S_x = S_dx + s_x + s1 @ C_dx + C_dx.T @ s1.T

                    return M_x, S_x

                return *_propogate(m_x, s_x), jnp.add(reward, jnp.squeeze(self.reward.compute_reward(m_x, s_x)[0]))

            val = jax.lax.fori_loop(0, self.agent_config.PLANNING_HORIZON, body_fun, init_val)

            m_x, s_x, reward = val

            return -reward

        loss = training_loss()

        # opt_hypers = objax.optimizer.Adam(self.controller.vars())
        # energy = objax.GradValues(self.training_loss, self.controller.vars())
        #
        # def train_op(en=energy, oh=opt_hypers):
        #     dE, E = en()
        #     oh(lr_adam, dE)
        #     return E
        #
        # self.optimizer = objax.Jit(objax.Function(train_op, self.controller.vars() + opt_hypers.vars()))
        #
        # for i in range(maxiter):
        #     self.optimizer()
        #
        # best_parameter_values = [jnp.array(param) for param in self.controller.vars()]
        # best_reward = self.compute_reward()
        #
        # for restart in range(restarts):
        #     self.controller.randomize()
        #
        #     for i in range(maxiter):
        #         self.optimizer()
        #     reward = self.compute_reward()
        #     if reward > best_reward:
        #         best_parameter_values = [jnp.array(param) for param in self.controller.vars()]
        #         best_reward = reward
        #
        # for i, param in enumerate(self.controller.vars()):
        #     param.assign(best_parameter_values[i])

    def optimise_models(self, maxiter=1000, restarts=1):
        """
        Optimize GP models
        """
        self.mgpr.optimize(maxiter=maxiter, restarts=restarts)
        # lengthscales = {}
        # variances = {}
        # noises = {}
        # for i, model in enumerate(self.mgpr.models):
        #     lengthscales["GP" + str(i)] = jnp.array(model.kernel.lengthscale)
        #     variances["GP" + str(i)] = jnp.array([jnp.array(model.kernel.variance)])
        #     noises["GP" + str(i)] = jnp.array([jnp.array(model.likelihood.variance)])
        #
        # print("-----Learned models------")
        # print("---Lengthscales---")
        # print(pd.DataFrame(data=lengthscales))
        # print("---Variances---")
        # print(pd.DataFrame(data=variances))
        # print("---Noises---")
        # print(pd.DataFrame(data=noises))

    def compute_action(self, x_m, train_state):
        return self.controller.apply(train_state.params, x_m, jnp.zeros([self.obs_dim, self.obs_dim]))

    def compute_reward(self):
        return -self.training_loss()

    @property
    def maximum_log_likelihood_objective(self):
        return -self.training_loss()

    def get_next_point(self, curr_obs_O, train_state, key):
        # do the usual act and all that
        action = self.controller.compute_action()

        # if num_iters // planning horizon then also do the optimisation stuff

        x_next = ""
        exe_path = ""
        return x_next, exe_path, curr_obs_O, train_state, None, key




