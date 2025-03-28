"""
Based off the following code "https://github.com/mathDR/jax-pilco/blob/main/pilco/models/pilco.py" and original paper
"""


from project_name.agents.agent_base import AgentBase

import jax.numpy as jnp
import jax
import jax.random as jrandom
from project_name.utils import update_obs_fn, update_obs_fn_teleport, get_f_mpc, get_f_mpc_teleport
from project_name import dynamics_models
from project_name.agents.PILCO import LinearController, get_PILCO_config, ExponentialReward
import optax
from flax.training.train_state import TrainState
import flax.linen as nn
import GPJax_AScannell as gpjaxas
from jaxtyping import Float, install_import_hook
from functools import partial

with install_import_hook("gpjax", "beartype.beartype"):
    import logging
    logging.getLogger('gpjax').setLevel(logging.WARNING)
    import gpjax


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
    It imports from MPC for ground truth creation, is that okay idk?
    """
    def __init__(self, env, env_params, config, key):
        super().__init__(env, env_params, config, key)
        self.agent_config = get_PILCO_config()

        # TODO add some import from folder check thingo
        self.dynamics_model = dynamics_models.MOSVGPGPJax(env, env_params, config, self.agent_config, key)

        self.obs_dim = len(self.env.observation_space(self.env_params).low)
        self.action_dim = self.env.action_space().shape[0]
        # TODO match this to the other rl main stuff

        self.controller = LinearController(self.obs_dim, self.action_dim, self.env.action_space().high)
        self.reward = ExponentialReward(self.obs_dim,
                                        w_init=lambda x, y: jnp.reshape(jnp.diag(jnp.array([2.0, 0.3])), (self.obs_dim, self.obs_dim)),
                                        t_init=lambda x, y: jnp.reshape(jnp.array([0.0, 0.0]), (1, self.obs_dim)))
        # TODO the above is hardcoded for pendulum

        self.tx = optax.adam(self.agent_config.POLICY_LR)

        if config.TELEPORT:
            self._update_fn = update_obs_fn_teleport
        else:
            self._update_fn = update_obs_fn

        self.m_init = jnp.reshape(jnp.array([0.0, 0.0]), (1, 2))
        self.S_init = jnp.diag(jnp.array([0.05, 0.01]))
        # TODO both above are for pendulum, can we generalise

    def create_train_state(self, init_data_x, init_data_y, key):
        train_state = {}

        key, _key = jrandom.split(key)
        train_state["dynamics_train_state"] = self.dynamics_model.create_train_state(init_data_x, init_data_y, _key)

        key, _key = jrandom.split(key)
        params = self.controller.init(_key,
                                      jnp.zeros((1, self.obs_dim)),
                                      jnp.zeros((self.obs_dim, self.obs_dim)))
        controller_train_state = TrainState.create(apply_fn=self.controller.apply, params=params, tx=self.tx)
        train_state["controller_train_state"] = controller_train_state

        key, _key = jrandom.split(key)
        params = self.reward.init(_key,
                                      jnp.zeros((1, self.obs_dim)),
                                      jnp.zeros((self.obs_dim, self.obs_dim)))
        reward_train_state = TrainState.create(apply_fn=self.reward.apply, params=params, tx=self.tx)
        train_state["reward_train_state"] = reward_train_state

        return train_state

    def pretrain_params(self, init_data_x, init_data_y, pretrain_data_x, pretrain_data_y, key):
        # optimisation for the dynamics model
        train_state = self.create_train_state(init_data_x, init_data_y, key)
        train_state["dynamics_train_state"] = self.dynamics_model.pretrain_params(init_data_x, init_data_y, pretrain_data_x, pretrain_data_y, key)

        # controller optimisation
        pretrain_data = gpjax.Dataset(pretrain_data_x, pretrain_data_y)
        train_state = self._optimise_policy(train_state, pretrain_data, key)

        return train_state

    def _optimise_policy(self, train_state, train_data, key, maxiter=10, restarts=5):  # original is 1000
        def training_loss(controller_params):
            # This is for tuning controller's parameters
            init_val = (self.m_init, self.S_init, 0.0)

            def _body_fun(v, unused):
                m_x, s_x, reward = v

                def _propogate(m_x, s_x):
                    m_u, s_u, c_xu = self.controller.apply(controller_params, m_x, s_x)

                    m = jnp.concatenate([m_x, m_u], axis=1)
                    s1 = jnp.concatenate([s_x, s_x @ c_xu], axis=1)
                    s2 = jnp.concatenate([jnp.transpose(s_x @ c_xu), s_u], axis=1)
                    s = jnp.concatenate([s1, s2], axis=0)

                    M_dx, S_dx, C_dx = self.dynamics_model.predict_on_noisy_inputs(m, s, train_state["dynamics_train_state"], train_data)
                    M_x = M_dx + m_x
                    S_x = S_dx + s_x + s1 @ C_dx + C_dx.T @ s1.T

                    return M_x, S_x

                return (*_propogate(m_x, s_x), jnp.add(reward, jnp.squeeze(self.reward.apply(train_state["reward_train_state"].params, m_x, s_x)[0]))), None

            # val = jax.lax.fori_loop(0, self.agent_config.PLANNING_HORIZON, _body_fun, init_val)
            val, _ = jax.lax.scan(_body_fun, init_val, None, self.agent_config.PLANNING_HORIZON)
            # TODO turn this into scan if it gets working

            m_x, s_x, reward = val

            return -reward

        # Optimisation step.
        def optimisation_step(init_train_state):
            def _step_fn(controller_train_state, unused):
                loss_val, grads = jax.value_and_grad(training_loss)(controller_train_state.params)
                new_controller_state = controller_train_state.apply_gradients(grads=grads)
                return new_controller_state, loss_val

            # Optimisation loop.
            new_controller_state, history = jax.lax.scan(_step_fn, init_train_state, None, maxiter)

            best_reward = -history[-1]

            return new_controller_state, best_reward

        # an initial optimisation
        new_controller_state, best_init_reward = optimisation_step(train_state["controller_train_state"])

        # do some vmap over some randomised params for the controller and run the above
        def randomise_restart(key):
            controller = LinearController(self.obs_dim, self.action_dim, self.env.action_space().high,
                                          w_init=nn.initializers.normal(stddev=1),
                                          b_init=nn.initializers.normal(stddev=1))
            # TODO a bit dodgy but may work for now
            params = controller.init(key,
                                      jnp.zeros((1, self.obs_dim)),
                                      jnp.zeros((self.obs_dim, self.obs_dim)))
            randomised_train_state = TrainState.create(apply_fn=self.controller.apply, params=params, tx=self.tx)
            new_params, best_reward = optimisation_step(randomised_train_state)

            return new_params, best_reward

        key, _key = jrandom.split(key)
        restart_key = jrandom.split(_key, restarts)
        randomised_controller_best_states, batch_rewards = jax.vmap(randomise_restart)(restart_key)

        all_rewards = jnp.concatenate((jnp.expand_dims(best_init_reward, axis=0), batch_rewards))
        all_controller_states = jax.tree_util.tree_map(lambda x, y: jnp.concatenate((jnp.expand_dims(x, axis=0), y)), new_controller_state, randomised_controller_best_states)

        best_reward_idx = jnp.argmax(all_rewards)
        new_controller_state = jax.tree_util.tree_map(lambda x: x[best_reward_idx], all_controller_states)

        train_state["controller_train_state"] = new_controller_state

        return train_state

    def compute_action(self, x_m, train_state):
        return self.controller.apply(train_state.params, x_m, jnp.zeros([self.obs_dim, self.obs_dim]))


    def get_next_point(self, curr_obs_O, train_state, train_data, step_idx, key):
        # do the usual act and all that
        action_1A = self.controller.apply(train_state["controller_train_state"].params, curr_obs_O[None, :], jnp.zeros((self.obs_dim, self.obs_dim)))[0]

        if (step_idx + 1) // self.agent_config.PLANNING_HORIZON:
            dynamics_train_state = self.dynamics_model.optimise_gp(train_data, train_state["dynamics_train_state"], key)

            kernel = gpjaxas.kernels.SeparateIndependent(
                [gpjaxas.kernels.SquaredExponential(lengthscales=dynamics_train_state[0]["kernel"]["lengthscales"][idx],
                                                    variance=dynamics_train_state[0]["kernel"]["variance"][idx]) for idx in
                 range(self.obs_dim)])
            # TODO dodgy fix for now assuming the kernels are the same

            train_state["dynamics_train_state"]["kernel"] = kernel.get_params()

            train_state = self._optimise_policy(train_state, train_data, key)

        x_next_OPA = jnp.concatenate((curr_obs_O, jnp.squeeze(action_1A, axis=0)), axis=-1)
        exe_path = jnp.expand_dims(curr_obs_O, axis=0)
        # TODO the above is a weird fix but may work

        assert jnp.allclose(curr_obs_O, x_next_OPA[:self.obs_dim]), "For rollout cases, we can only give queries which are from the current state"
        # TODO can we jax the assertion?

        return x_next_OPA, exe_path, curr_obs_O, train_state, None, key

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, start_obs, start_env_state, train_state, sep_data, key):
        # TODO sort this out at some point
        return




