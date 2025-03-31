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
from project_name import utils
from flax import nnx
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
        self.S_init = jnp.diag(jnp.array([0.03, 0.01]))
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

        data = gpjax.Dataset(pretrain_data_x, pretrain_data_y)
        train_state["dynamics_train_state"] = self._optimise_gp(data, train_state, key)

        # controller optimisation
        pretrain_data = gpjax.Dataset(pretrain_data_x, pretrain_data_y)
        train_state = self._optimise_policy(train_state, pretrain_data, key)

        return train_state

    def get_batch(self, train_data: Dataset, batch_size: int, key: KeyArray) -> Dataset:
        """Batch the data into mini-batches. Sampling is done with replacement.

        Args:
            train_data (Dataset): The training dataset.
            batch_size (int): The batch size.
            key (KeyArray): The random key to use for the batch selection.

        Returns
        -------
            Dataset: The batched dataset.
        """
        x, y, n = train_data.X, train_data.y, train_data.n

        # Subsample mini-batch indices with replacement.
        indices = jrandom.choice(key, n, (batch_size,), replace=True)

        return Dataset(X=x[indices], y=y[indices])

    # @partial(jax.jit, static_argnums=(0,))
    def _optimise_gp(self, opt_data, train_state, key):
        key, _key = jrandom.split(key)
        data = self.dynamics_model._adjust_dataset(opt_data)
        q = self.dynamics_model.variational_posterior_builder(data.n)

        graphdef, state = nnx.split(q)
        q = nnx.merge(graphdef, train_state["dynamics_train_state"]["train_state"])

        # schedule = optax.warmup_cosine_decay_schedule(init_value=0.0,
        #                                               peak_value=0.02,
        #                                               warmup_steps=75,
        #                                               decay_steps=2000,
        #                                               end_value=0.001)
        # # TODO idk if we need the above
        #
        # opt_posterior, _ = gpjax.fit(model=q,
        #                              objective=lambda p, d: -gpjax.objectives.elbo(p, d),
        #                              train_data=data,
        #                              optim=optax.adam(learning_rate=self.agent_config.GP_LR),
        #                              # optim=optax.adam(learning_rate=schedule),
        #                              num_iters=self.agent_config.TRAIN_GP_NUM_ITERS,
        #                              batch_size=128,
        #                              safe=True,
        #                              key=_key,
        #                              verbose=False)

        train_data = opt_data
        optim = optax.adam(learning_rate=self.agent_config.GP_LR)
        objective = lambda dp, cp, rp, d: -self._training_loss(dp, cp, rp, d)
        num_iters = self.agent_config.TRAIN_GP_NUM_ITERS
        batch_size = 128
        unroll = 1

        graphdef, params, *static_state = nnx.split(q, Parameter, ...)

        # Parameters bijection to unconstrained space
        params = transform(params, DEFAULT_BIJECTION, inverse=True)

        # Loss definition
        def loss(params: nnx.State, batch: Dataset) -> ScalarFloat:
            params = transform(params, DEFAULT_BIJECTION)
            model = nnx.merge(graphdef, params, *static_state)
            return objective(model, train_state["controller_train_state"].params, train_state["reward_train_state"].params, batch)
            # return objective(params, train_state["controller_train_state"].params, train_state["reward_train_state"].params, batch)

        # Initialise optimiser state.
        opt_state = optim.init(params)

        # Mini-batch random keys to scan over.
        iter_keys = jrandom.split(key, num_iters)

        # Optimisation step.
        def step(carry, key):
            params, opt_state = carry

            if batch_size != -1:
                batch = self.get_batch(train_data, batch_size, key)
            else:
                batch = train_data

            loss_val, loss_gradient = jax.value_and_grad(loss, argnums=0)(params, batch)
            updates, opt_state = optim.update(loss_gradient, opt_state, params)
            params = optax.apply_updates(params, updates)

            carry = params, opt_state
            return carry, loss_val

        # Optimisation loop.
        (params, _), history = jax.lax.scan(step, (params, opt_state), (iter_keys), unroll=unroll)

        # Parameters bijection to constrained space
        params = transform(params, DEFAULT_BIJECTION)

        # Reconstruct model
        opt_posterior = nnx.merge(graphdef, params, *static_state)

        graphdef, state = nnx.split(opt_posterior)

        return {"train_state": state}

    def _training_loss(self, dynamics_params, controller_params, reward_params, train_data):
        # This is for tuning controller's parameters
        init_val = (self.m_init, self.S_init, 0.0)

        def _body_fun(v, unused):
            m_x, s_x, reward = v

            def _propagate(m_x, s_x):
                m_u, s_u, c_xu = self.controller.apply(controller_params, m_x, s_x)

                m = jnp.concatenate([m_x, m_u], axis=1)
                s1 = jnp.concatenate([s_x, s_x @ c_xu], axis=1)
                s2 = jnp.concatenate([jnp.transpose(s_x @ c_xu), s_u], axis=1)
                s = jnp.concatenate([s1, s2], axis=0)

                M_dx, S_dx, C_dx = self.dynamics_model.predict_on_noisy_inputs(m, s, dynamics_params, train_data)
                M_x = M_dx + m_x
                S_x = S_dx + s_x + s1 @ C_dx + C_dx.T @ s1.T

                # new_M_dx, new_S_dx = self.dynamics_model.get_post_mu_fullcov2(m, s, dynamics_params, train_data)
                # M_x = new_M_dx + m_x
                # S_x = new_S_dx[0]
                # # TODO above is kinda dodgy, does it work?

                return M_x, S_x

            return (*_propagate(m_x, s_x), jnp.add(reward, jnp.squeeze(
                self.reward.apply(reward_params, m_x, s_x)[0]))), None

        val, _ = jax.lax.scan(_body_fun, init_val, None, self.agent_config.PLANNING_HORIZON)
        m_x, s_x, reward = val

        return -reward

    @partial(jax.jit, static_argnums=(0,))
    def _optimise_policy(self, train_state, train_data, key, maxiter=10, restarts=5):  # original is 1000
        # Optimisation step.
        def optimisation_step(init_train_state):
            def _step_fn(controller_train_state, unused):
                q = self.dynamics_model.variational_posterior_builder(train_data.n)
                graphdef, params, *static_state = nnx.split(q, Parameter, ...)
                model = nnx.merge(graphdef, train_state["dynamics_train_state"]["train_state"], *static_state)
                loss_val, grads = jax.value_and_grad(self._training_loss, argnums=1)(model,
                                                                                     controller_train_state.params,
                                                                                     train_state["reward_train_state"].params,
                                                                                     train_data)
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

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, x_m, train_state):
        return self.controller.apply(train_state.params, x_m, jnp.zeros([self.obs_dim, self.obs_dim]))

    def empty_func(self, train_state):
        return train_state

    def get_next_point(self, curr_obs_O, train_state, train_data, step_idx, key):
        # do the usual act and all that
        action_1A = self.controller.apply(train_state["controller_train_state"].params, curr_obs_O[None, :], jnp.zeros((self.obs_dim, self.obs_dim)))[0]

        if (step_idx + 1) % self.agent_config.PLANNING_HORIZON == 0:
            train_state["dynamics_train_state"] = self._optimise_gp(train_data, train_state, key)
            # dynamics_train_state = self.dynamics_model.optimise_gp(train_data, train_state["dynamics_train_state"], key)
            # train_state["dynamics_train_state"] = dynamics_train_state
            # with jax.disable_jit(disable=False):
            train_state = self._optimise_policy(train_state, train_data, key)

        x_next_OPA = jnp.concatenate((curr_obs_O, jnp.squeeze(action_1A, axis=0)), axis=-1)
        exe_path = {"exe_path_x": jnp.zeros((1, 10, 3)),
                      "exe_path_y": jnp.zeros((1, 10, 2))}
        # TODO the above is a bad fix for now

        assert jnp.allclose(curr_obs_O, x_next_OPA[:self.obs_dim]), "For rollout cases, we can only give queries which are from the current state"
        # TODO can we jax the assertion?

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




