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
from jaxtyping import Float, install_import_hook
from functools import partial
from project_name import utils
from flax import nnx
from jax.experimental import checkify
from gpjax.parameters import (
    DEFAULT_BIJECTION,
    Parameter,
    transform,
)
from gpjax.dataset import Dataset
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)
import sys

with install_import_hook("gpjax", "beartype.beartype"):
    import logging
    logging.getLogger('gpjax').setLevel(logging.WARNING)
    import gpjax


class PILCOAgent(AgentBase):

    def __init__(self, env, env_params, config, key):
        super().__init__(env, env_params, config, key)
        self.agent_config = get_PILCO_config()

        # TODO add some import from folder check thingo
        self.dynamics_model = dynamics_models.MOSVGP(env, env_params, config, self.agent_config, key)

        self.obs_dim = len(self.env.observation_space(self.env_params).low)
        self.action_dim = self.env.action_space(env_params).shape[0]
        # TODO match this to the other rl main stuff

        self.controller = LinearController(self.obs_dim, self.action_dim, self.env.action_space(env_params).high)
        self.reward = ExponentialReward(self.obs_dim,
                                        w_init=lambda x, y: jnp.reshape(jnp.diag(jnp.array([2.0, 0.3])), (self.obs_dim, self.obs_dim)),
                                        t_init=lambda x, y: jnp.reshape(jnp.array([0.0, 0.0]), (1, self.obs_dim)))
        # TODO the above is hardcoded for pendulum

        self.tx = optax.adam(self.agent_config.POLICY_LR)

        # self.tx = optax.chain(optax.clip_by_global_norm(self.agent_config.MAX_GRAD_NORM),
        #                       optax.adam(self.agent_config.POLICY_LR))

        if config.TELEPORT:
            self._update_fn = update_obs_fn_teleport
        else:
            self._update_fn = update_obs_fn

    def create_train_state(self, init_data, key):
        train_state = {}

        key, _key = jrandom.split(key)
        train_state["dynamics_train_state"] = self.dynamics_model.create_train_state(init_data, _key)

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

        self.m_init = init_data.X[0:1, 0:self.obs_dim]
        self.S_init = jnp.diag(jnp.ones(self.obs_dim) * 0.1)

        return train_state

    def pretrain_params(self, init_data, pretrain_data, key):
        # optimisation for the dynamics model
        key, _key = jrandom.split(key)
        train_state = self.create_train_state(init_data, _key)

        key, _key = jrandom.split(key)
        # train_state["dynamics_train_state"] = self.dynamics_model.pretrain_params(init_data, pretrain_data, _key)
        train_state["dynamics_train_state"] = self.dynamics_model.pretrain_params(init_data, init_data, _key)

        # controller optimisation
        key, _key = jrandom.split(key)
        with jax.disable_jit(disable=False):
            train_state = self._optimise_policy(train_state, init_data, _key)

        return train_state

    # def get_batch(self, train_data: Dataset, batch_size: int, key: KeyArray) -> Dataset:
    #     """Batch the data into mini-batches. Sampling is done with replacement.
    #
    #     Args:
    #         train_data (Dataset): The training dataset.
    #         batch_size (int): The batch size.
    #         key (KeyArray): The random key to use for the batch selection.
    #
    #     Returns
    #     -------
    #         Dataset: The batched dataset.
    #     """
    #     x, y, n = train_data.X, train_data.y, train_data.n
    #
    #     # Subsample mini-batch indices with replacement.
    #     indices = jrandom.choice(key, n, (batch_size,), replace=True)
    #
    #     return Dataset(X=x[indices], y=y[indices])
    #
    # # @partial(jax.jit, static_argnums=(0,))
    # def _optimise_gp(self, opt_data, train_state, key):
    #     key, _key = jrandom.split(key)
    #     data = self.dynamics_model._adjust_dataset(opt_data)
    #     q = self.dynamics_model.variational_posterior_builder(data.n)
    #
    #     graphdef, state = nnx.split(q)
    #     q = nnx.merge(graphdef, train_state["dynamics_train_state"]["train_state"])
    #
    #     schedule = optax.warmup_cosine_decay_schedule(init_value=0.0,
    #                                                   peak_value=0.02,
    #                                                   warmup_steps=75,
    #                                                   decay_steps=2000,
    #                                                   end_value=0.001)
    #     # TODO idk if we need the lr scheduler or not
    #
    #     opt_posterior, _ = gpjax.fit(model=q,
    #                                  objective=lambda p, d: -gpjax.objectives.elbo(p, d),
    #                                  train_data=data,
    #                                  # optim=optax.adam(learning_rate=self.agent_config.GP_LR),
    #                                  optim=optax.adam(learning_rate=schedule),
    #                                  num_iters=self.agent_config.TRAIN_GP_NUM_ITERS,
    #                                  batch_size=128,
    #                                  safe=True,
    #                                  key=_key,
    #                                  verbose=False)
    #
    #     # train_data = opt_data
    #     # optim = optax.adam(learning_rate=self.agent_config.GP_LR)
    #     # objective = lambda dp, cp, rp, d: -self._training_loss(dp, cp, rp, d)
    #     # num_iters = self.agent_config.TRAIN_GP_NUM_ITERS
    #     # batch_size = 128
    #     # unroll = 1
    #     #
    #     # graphdef, params, *static_state = nnx.split(q, Parameter, ...)
    #     #
    #     # # Parameters bijection to unconstrained space
    #     # params = transform(params, DEFAULT_BIJECTION, inverse=True)
    #     #
    #     # # Loss definition
    #     # def loss(params: nnx.State, batch: Dataset) -> ScalarFloat:
    #     #     params = transform(params, DEFAULT_BIJECTION)
    #     #     model = nnx.merge(graphdef, params, *static_state)
    #     #     return objective(model, train_state["controller_train_state"].params, train_state["reward_train_state"].params, batch)
    #     #     # return objective(params, train_state["controller_train_state"].params, train_state["reward_train_state"].params, batch)
    #     #
    #     # # Initialise optimiser state.
    #     # opt_state = optim.init(params)
    #     #
    #     # # Mini-batch random keys to scan over.
    #     # iter_keys = jrandom.split(key, num_iters)
    #     #
    #     # # Optimisation step.
    #     # def step(carry, key):
    #     #     params, opt_state = carry
    #     #
    #     #     if batch_size != -1:
    #     #         batch = self.get_batch(train_data, batch_size, key)
    #     #     else:
    #     #         batch = train_data
    #     #
    #     #     loss_val, loss_gradient = jax.value_and_grad(loss, argnums=0)(params, batch)
    #     #     updates, opt_state = optim.update(loss_gradient, opt_state, params)
    #     #     params = optax.apply_updates(params, updates)
    #     #
    #     #     carry = params, opt_state
    #     #     return carry, loss_val
    #     #
    #     # # Optimisation loop.
    #     # (params, _), history = jax.lax.scan(step, (params, opt_state), (iter_keys), unroll=unroll)
    #     #
    #     # # Parameters bijection to constrained space
    #     # params = transform(params, DEFAULT_BIJECTION)
    #     #
    #     # # Reconstruct model
    #     # opt_posterior = nnx.merge(graphdef, params, *static_state)
    #
    #     graphdef, state = nnx.split(opt_posterior)
    #
    #     return {"train_state": state}
    #
    # def training_loss_old(self, dynamics_params, controller_params, reward_params, train_data):
    #     # This is for tuning controller's parameters
    #     init_val = (self.m_init, self.S_init, 0.0)
    #
    #     def _body_fun(v, unused):
    #         m_x, s_x, reward = v
    #
    #         def _propagate(m_x, s_x):
    #             m_u, s_u, c_xu = self.controller.apply(controller_params, m_x, s_x)
    #
    #             m = jnp.concatenate([m_x, m_u], axis=1)
    #             s1 = jnp.concatenate([s_x, s_x @ c_xu], axis=1)
    #             s2 = jnp.concatenate([jnp.transpose(s_x @ c_xu), s_u], axis=1)
    #             s = jnp.concatenate([s1, s2], axis=0)
    #
    #             # m = jnp.expand_dims(jnp.array(((0, 0, 0.00035808))), axis=0)
    #             # s = jnp.array(((3.00000000e-02, 0.00000000e+00, 5.42722945e-05),
    #             #                    (0.00000000e+00, 1.00000000e-02, 2.36281741e-05),
    #             #                    (5.42722945e-05, 2.36281741e-05, 1.54011792e-07)))
    #             #
    #             # train_data_x = jnp.array(((-0.0666826,   0.87475473,  0.87598572),
    #             #                           (0.19413371,  0.42402982, -0.28221906),
    #             #                           (-0.84082947,  0.94610294, -0.90515271),
    #             #                           (0.51090783,  0.4784161,   0.12383541),
    #             #                           (-0.96840102,  0.54481848, -0.10317754),
    #             #                           (0.41205937,  0.65534962,  0.16859988),
    #             #                           (0.60086796, -0.47660882, -0.0124152),
    #             #                           (0.34696807, -0.28261206,  0.14367731),
    #             #                           (-0.32821263,  0.42114675, -0.82885184),
    #             #                           (0.43397144,  0.94669633,  0.49274196)))
    #             #
    #             # train_data_y = jnp.array(((0.11307741,  0.01335315),
    #             #                           (0.05947867,  0.04311458),
    #             #                           (0.11041655, -0.0788934),
    #             #                           (0.07343471,  0.09833879),
    #             #                           (0.06769279, -0.01316055),
    #             #                           (0.09573068,  0.09651736),
    #             #                           (-0.0494007,   0.08861665),
    #             #                           (-0.02471377,  0.08851058),
    #             #                           (0.03942452, -0.11150727),
    #             #                           (0.13457052,  0.05330367)))
    #
    #             M_dx, S_dx, C_dx = self.dynamics_model.predict_on_noisy_inputs(m, s, dynamics_params, train_data)
    #             M_x = M_dx + m_x
    #             S_x = S_dx + s_x + s1 @ C_dx + C_dx.T @ s1.T
    #
    #             # new_M_dx, new_S_dx = self.dynamics_model.get_post_mu_fullcov2(m, s, dynamics_params, train_data)
    #             # M_x = new_M_dx + m_x
    #             # S_x = new_S_dx[0]
    #             # # TODO above is kinda dodgy, does it work?
    #
    #             return M_x, S_x
    #
    #         return (*_propagate(m_x, s_x), jnp.add(reward, jnp.squeeze(
    #             self.reward.apply(reward_params, m_x, s_x)[0]))), None
    #
    #     val, _ = jax.lax.scan(_body_fun, init_val, None, self.agent_config.PLANNING_HORIZON)
    #     m_x, s_x, reward = val
    #
    #     return -reward

    # @partial(jax.jit, static_argnums=(0,))
    def _optimise_policy(self, train_state, train_data, key, maxiter=30, restarts=5):  # original is 1000
        def training_loss(controller_params, dynamics_params, reward_params, loss_train_data):
            # This is for tuning controller's parameters
            init_val = (self.m_init, self.S_init, jnp.zeros((1,)))

            def _propagate_step(runner_state, unused):
                m_x, s_x, reward = runner_state

                m_u, s_u, c_xu = self.controller.apply(controller_params, m_x, s_x)

                m = jnp.concatenate([m_x, m_u], axis=1)
                s1 = jnp.concatenate([s_x, s_x @ c_xu], axis=1)
                s2 = jnp.concatenate([jnp.transpose(s_x @ c_xu), s_u], axis=1)
                s = jnp.concatenate([s1, s2], axis=0)

                M_dx, S_dx, C_dx = self.dynamics_model.predict_on_noisy_inputs(m, s,
                                                                               dynamics_params,
                                                                               loss_train_data)
                M_x = M_dx + m_x
                S_x = S_dx + s_x + s1 @ C_dx + C_dx.T @ s1.T

                new_reward = reward + jnp.squeeze(self.reward.apply(reward_params, m_x, s_x)[0], axis=0)
                # TODO rewards get way too large, almost becoming much larger than 1
                # TODO also there becomes nans in the prediction, but it seems to be accurate to original
                # TODO what is causing this to get too large? It appears driven by data selection, what are the data issues that cause it?

                return (M_x, S_x, new_reward), new_reward

            # val = jax.lax.fori_loop(0, self.agent_config.PLANNING_HORIZON, _propagate_step, init_val)
            val, _ = jax.lax.scan(_propagate_step, init_val, None, self.agent_config.PLANNING_HORIZON)

            m_x, s_x, reward = val

            return -jnp.squeeze(reward)

        # Optimisation step.
        def optimisation_step(init_train_state):
            def _step_fn(train_state, unused):
                loss_val, grads = jax.value_and_grad(training_loss)(train_state["controller_train_state"].params,
                                                                    train_state["dynamics_train_state"],
                                                                    train_state["reward_train_state"].params,
                                                                    train_data)
                new_controller_train_state = train_state["controller_train_state"].apply_gradients(grads=grads)
                train_state["controller_train_state"] = new_controller_train_state
                return train_state, loss_val

            # Optimisation loop.
            new_train_state, history = jax.lax.scan(_step_fn, init_train_state, None, maxiter)

            best_reward = -history[-1]

            return new_train_state, best_reward

        # randomise controller for restarts
        def randomise_controller(key):
            controller = LinearController(self.obs_dim, self.action_dim, self.env.action_space(self.env_params).high,
                                          w_init=nn.initializers.normal(stddev=1),
                                          b_init=nn.initializers.normal(stddev=1))
            # TODO a bit dodgy but may work for now
            params = controller.init(key,
                                     jnp.zeros((1, self.obs_dim)),
                                     jnp.zeros((self.obs_dim, self.obs_dim)))
            randomised_controller_state = TrainState.create(apply_fn=self.controller.apply, params=params, tx=self.tx)
            return randomised_controller_state

        key, _key = jrandom.split(key)
        restart_key = jrandom.split(_key, restarts)
        randomised_controller = jax.vmap(randomise_controller)(restart_key)
        # join randomised controller with our current known value
        controller_states = jax.tree_util.tree_map(lambda x, y: jnp.concatenate((jnp.expand_dims(x, axis=0), y)),
                                                       train_state["controller_train_state"],
                                                       randomised_controller)

        # vmap over some randomised params for the controller and run the above
        def randomise_restart(randomised_controller_state, overall_train_state):

            overall_train_state["controller_train_state"] = randomised_controller_state
            new_params, best_reward = optimisation_step(overall_train_state)

            return new_params["controller_train_state"], best_reward

        # perform optermisation vmapped over all possible controller states
        randomised_controller_best_states, all_rewards = jax.vmap(randomise_restart, in_axes=(0, None))(controller_states,
                                                                                                          train_state)
        # sometimes training returns nans, unsure why but the below prevents
        best_reward_idx = jnp.nanargmax(all_rewards)
        # extract the best controller_state and use it for next batch of steps
        new_controller_state = jax.tree_util.tree_map(lambda x: x[best_reward_idx], randomised_controller_best_states)
        train_state["controller_train_state"] = new_controller_state

        return train_state

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, x_m, train_state):
        return self.controller.apply(train_state.params, x_m, jnp.zeros([self.obs_dim, self.obs_dim]))

    # @partial(jax.jit, static_argnums=(0,))  # can't jit due to conditional, not sure what is quicker tho
    def get_next_point(self, curr_obs_O, train_state, train_data, step_idx, key):
        # do the usual act and all that
        action_1A = self.controller.apply(train_state["controller_train_state"].params, curr_obs_O[None, :], jnp.zeros((self.obs_dim, self.obs_dim)))[0]

        if (step_idx + 1) % self.agent_config.PLANNING_HORIZON == 0:
            with jax.disable_jit(disable=False):
                train_state["dynamics_train_state"] = self.dynamics_model.optimise_gp(train_data, train_state["dynamics_train_state"], key)
                train_state = self._optimise_policy(train_state, train_data, key)

        x_next_OPA = jnp.concatenate((curr_obs_O, jnp.squeeze(action_1A, axis=0)), axis=-1)
        exe_path = {"exe_path_x": jnp.zeros((1, 10, 3)),
                      "exe_path_y": jnp.zeros((1, 10, 2))}
        # TODO the above is a bad fix for now

        checkify.check(jnp.allclose(curr_obs_O, x_next_OPA[:self.obs_dim]),
                       "For rollout cases, we can only give queries which are from the current state")


        return x_next_OPA, exe_path, curr_obs_O, train_state, None, key

    @partial(jax.jit, static_argnums=(0,))
    def _compute_returns(self, rewards):  # MUST BE SHAPE batch, horizon as polyval uses shape horizon, batch
        return jnp.polyval(rewards.T, self.agent_config.DISCOUNT_FACTOR)
        # TODO is this correct for PILCO?

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, start_obs, start_env_state, train_state, sep_data, key):
        # train_data = gpjax.Dataset(sep_data[0], sep_data[1])

        def _env_step(env_runner_state, unused):
            obs_O, env_state, key = env_runner_state
            key, _key = jrandom.split(key)
            action_1A = self.controller.apply(train_state["controller_train_state"].params, obs_O[None, :], jnp.zeros((self.obs_dim, self.obs_dim)))[0]
            action_A = jnp.squeeze(action_1A, axis=0)
            key, _key = jrandom.split(key)
            nobs_O, new_env_state, reward, done, info = self.env.step(_key, env_state, action_A, self.env_params)
            return (nobs_O, new_env_state, key), (nobs_O, reward, action_A)

        key, _key = jrandom.split(key)
        _, (nobs_SO, real_rewards_S, real_actions_SA) = jax.lax.scan(_env_step, (start_obs, start_env_state, _key),
                                                                     None, self.env_params.horizon)

        real_obs_SP1O = jnp.concatenate((jnp.expand_dims(start_obs, axis=0), nobs_SO))
        real_returns_1 = self._compute_returns(jnp.expand_dims(real_rewards_S, axis=0))
        real_path_x_SOPA = jnp.concatenate((real_obs_SP1O[:-1], real_actions_SA), axis=-1)
        real_path_y_SO = real_obs_SP1O[1:] - real_obs_SP1O[:-1]
        key, _key = jrandom.split(key)
        # real_path_y_hat_SO = self.make_postmean_func2()(real_path_x_SOPA, None, None, train_state,
        #                                                 train_data, _key)
        real_path_y_hat_SO = real_path_y_SO
        # TODO dodgy fix for now but should sort it out
        mse = 0.5 * jnp.mean(jnp.sum(jnp.square(real_path_y_SO - real_path_y_hat_SO), axis=1))

        return (utils.RealPath(x=real_path_x_SOPA, y=real_path_y_SO, y_hat=real_path_y_hat_SO),
                jnp.squeeze(real_returns_1), jnp.mean(real_returns_1), jnp.std(real_returns_1), jnp.mean(mse))




