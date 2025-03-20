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
from project_name.agents.MPC import get_MPC_config
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


class PILCOAgent(AgentBase):
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
    """
    def __init__(self, env, env_params, config, key):
        super().__init__(env, env_params, config, key)
        self.agent_config = get_MPC_config()

        # TODO add some import from folder check thingo
        self.dynamics_model = dynamics_models.MOGP(env, env_params, config, self.agent_config, key)

        self.controller = LinearController()

        self.obs_dim = len(self.env.observation_space(self.env_params).low)
        self.action_dim = self.env.action_space().shape[0]
        # TODO match this to the other rl main stuff

        if config.TELEPORT:
            self._update_fn = update_obs_fn_teleport
        else:
            self._update_fn = update_obs_fn

    def create_train_state(self, init_data_x, init_data_y, key):
        return self.dynamics_model.create_train_state(init_data_x, init_data_y, key)

    def pretrain_params(self, init_data_x, init_data_y, key):
        # TODO add some optimisation function for the model and policy

        return self.dynamics_model.pretrain_params(init_data_x, init_data_y, key)

    def training_loss(self):
        # This is for tuning controller's parameters
        reward = self.predict(self.m_init, self.S_init, self.horizon)[2]
        return -reward

    def optimise_models(self, maxiter=1000, restarts=1):
        """
        Optimize GP models
        """
        self.mgpr.optimize(maxiter=maxiter, restarts=restarts)
        # ToDo: only do this if verbosity is large enough
        lengthscales = {}
        variances = {}
        noises = {}
        for i, model in enumerate(self.mgpr.models):
            lengthscales["GP" + str(i)] = jnp.array(model.kernel.lengthscale)
            variances["GP" + str(i)] = jnp.array([jnp.array(model.kernel.variance)])
            noises["GP" + str(i)] = jnp.array([jnp.array(model.likelihood.variance)])

        print("-----Learned models------")
        print("---Lengthscales---")
        print(pd.DataFrame(data=lengthscales))
        print("---Variances---")
        print(pd.DataFrame(data=variances))
        print("---Noises---")
        print(pd.DataFrame(data=noises))

    def optimise_policy(self, maxiter=1000, restarts=1):
        """
        Optimize controller's parameter's
        """
        lr_adam = 0.1
        if not self.optimizer:
            opt_hypers = objax.optimizer.Adam(self.controller.vars())
            energy = objax.GradValues(self.training_loss, self.controller.vars())

            def train_op(en=energy, oh=opt_hypers):
                dE, E = en()
                oh(lr_adam, dE)
                return E

            self.optimizer = objax.Jit(
                objax.Function(
                    train_op,
                    self.controller.vars() + opt_hypers.vars(),
                )
            )

            for i in range(maxiter):
                self.optimizer()
        else:
            for i in range(maxiter):
                self.optimizer()

        best_parameter_values = [jnp.array(param) for param in self.controller.vars()]
        best_reward = self.compute_reward()

        for restart in range(restarts):
            self.controller.randomize()

            for i in range(maxiter):
                self.optimizer()
            reward = self.compute_reward()
            if reward > best_reward:
                best_parameter_values = [
                    jnp.array(param) for param in self.controller.vars()
                ]
                best_reward = reward

        for i, param in enumerate(self.controller.vars()):
            param.assign(best_parameter_values[i])

    def compute_action(self, x_m):
        return self.controller.compute_action(
            x_m, jnp.zeros([self.state_dim, self.state_dim])
        )[0]

    def predict(self, m_x, s_x, n):
        init_val = (m_x, s_x, 0.0)

        def body_fun(i, v):
            m_x, s_x, reward = v
            return (
                *self.propagate(m_x, s_x),
                jnp.add(reward, jnp.squeeze(self.reward.compute_reward(m_x, s_x)[0])),
            )

        val = fori_loop(0, n, body_fun, init_val)

        m_x, s_x, reward = val
        return m_x, s_x, reward

    def propagate(self, m_x, s_x):
        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        m = jnp.concatenate([m_x, m_u], axis=1)
        s1 = jnp.concatenate([s_x, s_x @ c_xu], axis=1)
        s2 = jnp.concatenate([jnp.transpose(s_x @ c_xu), s_u], axis=1)
        s = jnp.concatenate([s1, s2], axis=0)

        M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)
        M_x = M_dx + m_x
        # TODO: cleanup the following line
        S_x = S_dx + s_x + s1 @ C_dx + C_dx.T @ s1.T

        # While-loop requires the shapes of the outputs to be fixed
        # M_x.set_shape([1, self.state_dim])
        # S_x.set_shape([self.state_dim, self.state_dim])
        return M_x, S_x

    def compute_reward(self):
        return -self.training_loss()

    @property
    def maximum_log_likelihood_objective(self):
        return -self.training_loss()

    def get_next_point(self, curr_obs_O, train_state, key):
        # do the usual act and all that

        # if num_iters // planning horizon then also do the optimisation stuff

        x_next = ""
        exe_path = ""
        return x_next, exe_path, curr_obs_O, train_state, None, key



def squash_sin(m, s, max_action=None):
    """
    Squashing function, passing the controls mean and variance
    through a sinus, as in gSin.m. The output is in [-max_action, max_action].
    IN: mean (m) and variance(s) of the control input, max_action
    OUT: mean (M) variance (S) and input-output (C) covariance of the squashed
         control input
    """
    k = jnp.shape(m)[1]
    if max_action is None:
        max_action = jnp.ones((1, k))  # squashes in [-1,1] by default
    else:
        max_action = max_action * jnp.ones((1, k))

    M = max_action * jnp.exp(-0.5 * jnp.diag(s)) * jnp.sin(m)

    lq = -0.5 * (jnp.diag(s)[:, None] + jnp.diag(s)[None, :])
    q = jnp.exp(lq)
    mT = jnp.transpose(m, (1, 0))
    S = (jnp.exp(lq + s) - q) * jnp.cos(mT - m) - (jnp.exp(lq - s) - q) * jnp.cos(
        mT + m
    )
    S = 0.5 * max_action * jnp.transpose(max_action, (1, 0)) * S

    C = max_action * objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection())(
        jnp.exp(-0.5 * jnp.diag(s)) * jnp.cos(m)
    )
    return M, S, C.reshape((k, k))


class LinearController(objax.Module):
    def __init__(self, state_dim, control_dim, max_action=1.0):
        objax.random.Generator(0)
        self.W = objax.TrainVar(objax.random.uniform((control_dim, state_dim)))
        self.b = objax.TrainVar(objax.random.uniform((1, control_dim)))
        self.max_action = max_action

    def compute_action(self, m, s, squash=True):
        """
        Simple affine action:  M <- W(m-t) - b
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        """

        WT = jnp.transpose(self.W.value, (1, 0))
        M = m @ WT + self.b.value  # mean output
        S = self.W.value @ s @ WT  # output variance
        V = WT  # input output covariance
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2
        return M, S, V

    def randomize(self):
        mean = 0
        sigma = 1
        self.W.assign(mean + sigma * objax.random.normal(self.W.shape))
        self.b.assign(mean + sigma * objax.random.normal(self.b.shape))
