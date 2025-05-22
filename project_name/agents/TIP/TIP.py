import jax.numpy as jnp
from project_name.agents.TIP import get_TIP_config
import jax
from functools import partial
import jax.random as jrandom
from project_name.agents.MPC import MPCAgent
from project_name import dynamics_models
from jaxtyping import Float, install_import_hook
from project_name import utils
from flax import nnx
from jax.experimental import checkify

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax


class TIPAgent(MPCAgent):

    def __init__(self, env, config, key):
        super().__init__(env, config, key)
        self.agent_config = get_TIP_config()

        # TODO add some import from folder check thingo
        self.dynamics_model = dynamics_models.MOGP(env, config, self.agent_config, key)
        # self.dynamics_model = dynamics_models.MOSVGP(env, config, self.agent_config, key)

    def make_postmean_func_const_key(self):
        def _postmean_fn(obs_O, action_A, env, unused2, train_state, train_data, key):
            obsacts_1OPA = jnp.expand_dims(jnp.concatenate((obs_O, action_A), axis=-1), axis=0)
            mu_1O = self.dynamics_model.get_post_mu_cov_samples(obsacts_1OPA, train_state, train_data, key, full_cov=False)
            mu_O = mu_1O.squeeze(axis=0)
            nobs_O = env.apply_delta_obs(obs_O, mu_O)

            return nobs_O, mu_O, env.get_state(nobs_O)
        return _postmean_fn

    @partial(jax.jit, static_argnums=(0, 1))
    def _optimise(self, f, x_test, curr_env_state, train_state, train_data, exe_path_BSOPA, key):
        curr_obs_O = x_test[:self.obs_dim]
        mean = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.action_dim))
        init_var_divisor = 4
        var = jnp.ones_like(mean) * ((self.env.action_space().high - self.env.action_space().low) / init_var_divisor) ** 2

        def _iter_iCEM2(iCEM2_runner_state, unused):  # TODO perhaps we can generalise this from above
            mean_SA, var_SA, prev_samples, prev_returns, key = iCEM2_runner_state
            key, _key = jrandom.split(key)
            samples_BSA = self._iCEM_generate_samples(_key,
                                                      self.agent_config.NUM_CANDIDATES,
                                                      self.agent_config.PLANNING_HORIZON,
                                                      mean_SA,
                                                      var_SA)

            key, _key = jrandom.split(key)
            batch_key = jrandom.split(_key, self.agent_config.NUM_CANDIDATES)
            acq = jax.vmap(self._evaluate_samples, in_axes=(None, None, None, None, None, 0, None, 0))(f,
                                                                                                       curr_obs_O,
                                                                                                       curr_env_state,
                                                                                                       train_state,
                                                                                                       (train_data.X, train_data.y),
                                                                                                        samples_BSA,
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
            samples_concat_BP1SA = jnp.concatenate((samples_BSA, prev_samples), axis=0)
            returns_concat_BP1 = jnp.concatenate((returns_B, prev_returns))

            # rank returns and chooses the top N_ELITES as the new mean and var
            elite_idx = jnp.argsort(returns_concat_BP1)[-self.agent_config.N_ELITES:]
            elites_ISA = samples_concat_BP1SA[elite_idx, ...]
            elite_returns_I = returns_concat_BP1[elite_idx]

            mean_SA = jnp.mean(elites_ISA, axis=0)
            var_SA = jnp.var(elites_ISA, axis=0)

            return (mean_SA, var_SA, elites_ISA, elite_returns_I, key), (samples_concat_BP1SA, returns_concat_BP1)

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

    @partial(jax.jit, static_argnums=(0, 1))
    def _evaluate_samples(self, f, obs_O, env_state, train_state, split_data, samples_S1, exe_path_BSOPA, key):
        train_data = gpjax.Dataset(split_data[0], split_data[1])

        # run a for loop planning basically
        def _run_planning_horizon2(runner_state, actions_A):  # TODO again can we generalise this from above to save rewriting things
            obs_O, env_state, key = runner_state
            obsacts_OPA = jnp.concatenate((obs_O, actions_A))
            key, _key = jrandom.split(key)
            nobs_O, delta_obs_O, next_env_state = f(obs_O, actions_A, self.env, env_state, train_state, train_data, _key)
            return (nobs_O, next_env_state, key), obsacts_OPA

        _, x_list_SOPA = jax.lax.scan(jax.jit(_run_planning_horizon2), (obs_O, env_state, key), samples_S1)

        # TODO this part is the acquisition function so should be generalised at some point rather than putting it here
        # get posterior covariance for x_set
        _, post_cov = self.dynamics_model.get_post_mu_fullcov(x_list_SOPA, train_state, train_data, full_cov=True)

        # get posterior covariance for all exe_paths, so this be a vmap probably
        def _get_sample_cov(x_list_SOPA, exe_path_SOPA, params):
            new_dataset = train_data + gpjax.Dataset(exe_path_SOPA["exe_path_x"], exe_path_SOPA["exe_path_y"])
            return self.dynamics_model.get_post_mu_fullcov(x_list_SOPA, params, new_dataset, full_cov=True)
        # TODO this is fairly slow as it feeds in a large amount of gp data to get the sample cov
        # TODO can we speed this up?

        _, samp_cov = jax.vmap(_get_sample_cov, in_axes=(None, 0, None))(x_list_SOPA, exe_path_BSOPA, train_state)

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

    # @partial(jax.jit, static_argnums=(0, 1, 6, 7))
    # def execute_mpc_next_point(self, f, obs, train_state, split_data, key, horizon, actions_per_plan):
    #     train_data = gpjax.Dataset(split_data[0], split_data[1])
    #
    #     adj_data = self.dynamics_model._adjust_dataset(train_data)
    #
    #     likelihood = self.dynamics_model.likelihood_builder(adj_data.n)
    #     posterior = self.dynamics_model.prior * likelihood
    #
    #     graphdef, state = nnx.split(posterior)
    #     opt_posterior = nnx.merge(graphdef, train_state["train_state"])
    #
    #     key, _key = jrandom.split(key)
    #     sample_func = dynamics_models.adj_sample_approx(opt_posterior, num_samples=1, train_data=adj_data, key=_key, num_features=500)
    #
    #     full_path, output, sample_returns = self.run_algorithm_on_f(sample_func, obs, train_state, train_data, key, horizon, actions_per_plan)
    #
    #     action = output[1]
    #
    #     exe_path = self.get_exe_path_crop(output[0], output[1])
    #
    #     return action, exe_path, output

    @partial(jax.jit, static_argnums=(0,))
    def get_next_point(self, curr_obs_O, curr_env_state, train_state, train_data, step_idx, key):
        key, _key = jrandom.split(key)
        batch_key = jrandom.split(_key, self.agent_config.ACQUISITION_SAMPLES)

        def sample_key_train_state(train_state, key):
            train_state["sample_key"] = key
            return train_state

        batch_train_state = jax.vmap(sample_key_train_state, in_axes=(None, 0))(train_state, batch_key)
        # TODO kind of dodgy fix to get samples for the posterior but is it okay?

        # idea here is to run a batch of MPC on different posterior functions, can we sample a batch of params?
        # so that we can just call the GP on these params in a VMAPPED setting
        output_traj, exe_path_BSOPA = jax.vmap(self.execute_mpc, in_axes=(None, None, None, 0, None, 0, None, None))(
            self.make_postmean_func_const_key(),
            # self.make_postmean_func(),
            curr_obs_O,
            curr_env_state,
            batch_train_state,
            (train_data.X, train_data.y),
            batch_key,
            self.env.horizon,
            self.agent_config.ACTIONS_PER_PLAN)
        # TODO can't seem to vmap dataset

        # add in some test values
        key, _key = jrandom.split(key)
        x_test = jnp.concatenate((curr_obs_O, self.env.action_space().sample(_key)))

        # now optimise the dynamics model with the x_test
        # take the exe_path_list that has been found with different posterior samples using iCEM
        # x_data and y_data are what ever you have currently
        key, _key = jrandom.split(key)
        x_next, acq_val = self._optimise(self.make_postmean_func(), x_test, curr_env_state, train_state, train_data, exe_path_BSOPA, _key)

        return x_next, exe_path_BSOPA, train_state, acq_val, key
