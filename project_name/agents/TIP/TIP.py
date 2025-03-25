import jax.numpy as jnp
from project_name.agents.TIP import get_TIP_config
import jax
from functools import partial
import jax.random as jrandom
from project_name.agents.MPC import MPCAgent
from project_name import dynamics_models
from jaxtyping import Float, install_import_hook
from project_name import utils

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax


class TIPAgent(MPCAgent):

    def __init__(self, env, env_params, config, key):
        super().__init__(env, env_params, config, key)
        self.agent_config = get_TIP_config()

        # TODO add some import from folder check thingo
        # self.dynamics_model = dynamics_models.MOGP(env, env_params, config, self.agent_config, key)
        self.dynamics_model = dynamics_models.MOGPGPJax(env, env_params, config, self.agent_config, key)

    def make_postmean_func_const_key(self):
        def _postmean_fn(x, unused1, unused2, train_state, train_data, key):
            mu = self.dynamics_model.get_post_mu_cov_samples(x, train_state, train_data, train_state["sample_key"], full_cov=False)
            return jnp.squeeze(mu, axis=0)
        return _postmean_fn

    @partial(jax.jit, static_argnums=(0, 3))
    def _optimise(self, train_state, train_data, f, exe_path_BSOPA, x_test, key):
        curr_obs_O = x_test[:self.obs_dim]
        mean = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.action_dim))  # TODO this may not be zero if there is alreayd an action sequence, should check this
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
            acq = jax.vmap(self._evaluate_samples, in_axes=(None, None, None, None, 0, None, 0))(train_state,
                                                                                                 (train_data.X, train_data.y),
                                                                                           f,
                                                                                           curr_obs_O,
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

    @partial(jax.jit, static_argnums=(0, 3))
    def _evaluate_samples(self, train_state, split_data, f, obs_O, samples_S1, exe_path_BSOPA, key):
        train_data = gpjax.Dataset(split_data[0], split_data[1])

        # run a for loop planning basically
        def _run_planning_horizon2(runner_state, actions_A):  # TODO again can we generalise this from above to save rewriting things
            obs_O, key = runner_state
            obsacts_OPA = jnp.concatenate((obs_O, actions_A), axis=-1)
            key, _key = jrandom.split(key)
            data_y_O = f(jnp.expand_dims(obsacts_OPA, axis=0), None, None, train_state, train_data, _key)
            nobs_O = self._update_fn(obsacts_OPA, data_y_O, self.env, self.env_params)
            return (nobs_O, key), obsacts_OPA

        _, x_list_SOPA = jax.lax.scan(jax.jit(_run_planning_horizon2), (obs_O, key), samples_S1)

        # TODO this part is the acquisition function so should be generalised at some point rather than putting it here
        # get posterior covariance for x_set
        _, post_cov = self.dynamics_model.get_post_mu_cov(x_list_SOPA, train_state, train_data, full_cov=True)

        # get posterior covariance for all exe_paths, so this be a vmap probably
        def _get_sample_cov(x_list_SOPA, exe_path_SOPA, params):
            # params["train_data_x"] = jnp.concatenate((params["train_data_x"], exe_path_SOPA["exe_path_x"]))
            # params["train_data_y"] = jnp.concatenate((params["train_data_y"], exe_path_SOPA["exe_path_y"]))
            new_dataset = train_data + gpjax.Dataset(exe_path_SOPA["exe_path_x"], exe_path_SOPA["exe_path_y"])
            return self.dynamics_model.get_post_mu_cov(x_list_SOPA, params, new_dataset, full_cov=True)
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

    partial(jax.jit, static_argnums=(0, 1, 6, 7))

    def _run_algorithm_on_f(self, f, start_obs_O, train_state, train_data, key, horizon, actions_per_plan):
        def _outer_loop(outer_loop_state, unused):
            init_obs_O, init_mean_SA, init_var_S1, init_shift_actions_BSA, key = outer_loop_state

            def _iter_iCEM(iCEM_iter_state, unused):
                mean_SA, var_SA, key = iCEM_iter_state

                # loops over the below and then takes trajectories to resample ICEM if not initial
                key, _key = jrandom.split(key)
                init_candidate_actions_BSA = self._iCEM_generate_samples(_key,
                                                                         self.agent_config.NUM_CANDIDATES,
                                                                         self.agent_config.PLANNING_HORIZON,
                                                                         mean_SA,
                                                                         var_SA)

                def _run_single_sample_planning_horizon(init_samples_S1, key):
                    def _run_single_timestep(runner_state, actions_A):
                        obs_O, key = runner_state
                        obsacts_OPA = jnp.concatenate((obs_O, actions_A))
                        key, _key = jrandom.split(key)


                        # data_y_O = f(jnp.expand_dims(obsacts_OPA, axis=0), self.env, self.env_params, train_state,
                        #              train_data, _key)

                        # mu, std = self.dynamics_model.get_post_mu_cov(jnp.expand_dims(obsacts_OPA, axis=0), train_state, train_data, full_cov=False)
                        # data_y_O = jnp.squeeze(mu, axis=0)

                        mu = self.dynamics_model.get_post_mu_cov_samples(jnp.expand_dims(obsacts_OPA, axis=0), train_state, train_data, train_state["sample_key"], full_cov=False)
                        data_y_O = jnp.squeeze(mu, axis=0)


                        nobs_O = self._update_fn(obsacts_OPA, data_y_O, self.env, self.env_params)
                        reward = self.env.reward_function(obsacts_OPA, nobs_O, self.env_params)
                        return (nobs_O, key), utils.MPCTransitionXY(obs=nobs_O,
                                                              action=actions_A,
                                                              reward=jnp.expand_dims(reward, axis=-1),
                                                              x=obsacts_OPA, y=data_y_O)

                    return jax.lax.scan(_run_single_timestep, (init_obs_O, key), init_samples_S1,
                                        self.agent_config.PLANNING_HORIZON)

                init_actions_BSA = jnp.concatenate((init_candidate_actions_BSA, init_shift_actions_BSA), axis=0)
                key, _key = jrandom.split(key)
                batch_key = jrandom.split(_key, self.agent_config.NUM_CANDIDATES + self.n_keep)
                _, planning_traj_BSX = jax.vmap(_run_single_sample_planning_horizon, in_axes=0)(init_actions_BSA,
                                                                                                batch_key)

                # compute return on the entire training list
                all_returns_B = self._compute_returns(jnp.squeeze(planning_traj_BSX.reward, axis=-1))

                # rank returns and chooses the top N_ELITES as the new mean and var
                elite_idx = jnp.argsort(all_returns_B)[-self.agent_config.N_ELITES:]
                elites_ISA = init_actions_BSA[elite_idx]

                new_mean_SA = jnp.mean(elites_ISA, axis=0)
                new_var_SA = jnp.var(elites_ISA, axis=0)

                mpc_transition_xy_BSX = utils.MPCTransitionXY(obs=planning_traj_BSX.obs[elite_idx],
                                                        action=planning_traj_BSX.action[elite_idx],
                                                        reward=planning_traj_BSX.reward[elite_idx],
                                                        x=planning_traj_BSX.x[elite_idx],
                                                        y=planning_traj_BSX.y[elite_idx])

                return (new_mean_SA, new_var_SA, key), mpc_transition_xy_BSX

            (best_mean_SA, best_var_SA, key), iCEM_traj_RISX = jax.lax.scan(_iter_iCEM,
                                                                            (init_mean_SA, init_var_S1, key),
                                                                            None,
                                                                            self.agent_config.iCEM_ITERS)

            iCEM_traj_minus_xy_RISX = utils.MPCTransition(obs=iCEM_traj_RISX.obs,
                                                    action=iCEM_traj_RISX.action,
                                                    reward=iCEM_traj_RISX.reward)
            iCEM_traj_minus_xy_BSX = jax.tree_util.tree_map(lambda x: jnp.reshape(x,
                                                                                  (x.shape[0] * x.shape[1],
                                                                                   x.shape[2], x.shape[3])),
                                                            iCEM_traj_minus_xy_RISX)

            # find the best sample from iCEM
            all_returns_B = self._compute_returns(jnp.squeeze(iCEM_traj_minus_xy_BSX.reward, axis=-1))
            best_sample_idx = jnp.argmax(all_returns_B)
            best_iCEM_traj_SX = jax.tree_util.tree_map(lambda x: x[best_sample_idx], iCEM_traj_minus_xy_BSX)
            # TODO unsure if this is necessary as the below could also work fine
            # best_iCEM_traj_SX = jax.tree_util.tree_map(lambda x: x[-1, 0], iCEM_traj_RISX)

            # take the number of actions of that plan and add to the existing plan
            planned_iCEM_traj_LX = jax.tree_util.tree_map(lambda x: x[:actions_per_plan], best_iCEM_traj_SX)

            # shift obs
            curr_obs_O = best_iCEM_traj_SX.obs[actions_per_plan - 1]

            # shift actions
            keep_indices = jnp.argsort(all_returns_B)[-self.n_keep:]
            short_shifted_actions_BSMLA = iCEM_traj_minus_xy_BSX.action[keep_indices, actions_per_plan:, :]

            # sample new actions and concat onto the "taken" actions
            key, _key = jrandom.split(key)
            new_actions_batch_LBA = self._action_space_multi_sample(actions_per_plan, _key)
            shifted_actions_BSA = jnp.concatenate((short_shifted_actions_BSMLA,
                                                   jnp.swapaxes(new_actions_batch_LBA, 0, 1)), axis=1)

            # remake the mean for iCEM
            end_mean_SA = jnp.concatenate(
                (best_mean_SA[actions_per_plan:], jnp.zeros((actions_per_plan, self.action_dim))))
            end_var_SA = (jnp.ones_like(end_mean_SA) * ((self.env.action_space().high - self.env.action_space().low)
                                                        / self.agent_config.INIT_VAR_DIVISOR) ** 2)

            return (curr_obs_O, end_mean_SA, end_var_SA, shifted_actions_BSA, key), utils.MPCTransitionXYR(
                obs=planned_iCEM_traj_LX.obs,
                action=planned_iCEM_traj_LX.action,
                reward=planned_iCEM_traj_LX.reward,
                x=iCEM_traj_RISX.x,
                y=iCEM_traj_RISX.y,
                returns=all_returns_B)

        outer_loop_steps = horizon // actions_per_plan

        init_mean_S1 = jnp.zeros((self.agent_config.PLANNING_HORIZON, self.env.action_space().shape[0]))
        init_var_S1 = (jnp.ones_like(init_mean_S1) * ((
                                                                  self.env.action_space().high - self.env.action_space().low) / self.agent_config.INIT_VAR_DIVISOR) ** 2)
        shift_actions_BSA = jnp.zeros(
            (self.n_keep, self.agent_config.PLANNING_HORIZON, self.action_dim))  # is this okay to add zeros?

        (_, _, _, _, key), overall_traj = jax.lax.scan(_outer_loop,
                                                       (start_obs_O, init_mean_S1, init_var_S1, shift_actions_BSA, key),
                                                       None, outer_loop_steps)

        overall_traj_minus_xyr_BLX = utils.MPCTransition(obs=overall_traj.obs, action=overall_traj.action,
                                                   reward=overall_traj.reward)
        flattened_overall_traj_SX = jax.tree_util.tree_map(lambda x: x.reshape(x.shape[0] * x.shape[1], -1),
                                                           overall_traj_minus_xyr_BLX)
        # TODO check this flattens correctly aka the batch of L steps merges into a contiguous S

        flatenned_path_x = overall_traj.x.reshape((-1, overall_traj.x.shape[-1]))
        flatenned_path_y = overall_traj.y.reshape((-1, overall_traj.y.shape[-1]))
        # TODO check this actually flattens, do we even want to fllaten this, unsure what shape even is

        joiner_SP1O = jnp.concatenate((jnp.expand_dims(start_obs_O, axis=0), flattened_overall_traj_SX.obs))
        return ((flatenned_path_x, flatenned_path_y),
                (joiner_SP1O, flattened_overall_traj_SX.action, flattened_overall_traj_SX.reward),
                overall_traj.returns)

    @partial(jax.jit, static_argnums=(0, 1, 6, 7))
    def _execute_mpc(self, f, obs, train_state, split_data, key, horizon, actions_per_plan):
        train_data = gpjax.Dataset(split_data[0], split_data[1])

        full_path, output, sample_returns = self._run_algorithm_on_f(f, obs, train_state, train_data, key, horizon,
                                                                    actions_per_plan)

        action = output[1]

        exe_path = self.get_exe_path_crop(output[0], output[1])

        return action, exe_path, output

    # @partial(jax.jit, static_argnums=(0,))
    def get_next_point(self, curr_obs, train_state, train_data, step_idx, key):
        key, _key = jrandom.split(key)
        batch_key = jrandom.split(_key, self.agent_config.ACQUISITION_SAMPLES)

        def sample_key_train_state(train_state, key):
            train_state["sample_key"] = key
            return train_state

        batch_train_state = jax.vmap(sample_key_train_state, in_axes=(None, 0))(train_state, batch_key)
        # TODO kind of dodgy fix to get samples for the posterior but is it okay?

        # idea here is to run a batch of MPC on different posterior functions, can we sample a batch of params?
        # so that we can just call the GP on these params in a VMAPPED setting
        _, exe_path_BSOPA, _ = jax.vmap(self._execute_mpc, in_axes=(None, None, 0, None, 0, None, None))(
            # self.make_postmean_func_const_key(),
            self.make_postmean_func(),
            curr_obs,
            batch_train_state,
            (train_data.X, train_data.y),
            batch_key,
            self.env_params.horizon,
            self.agent_config.ACTIONS_PER_PLAN)
        # TODO can't seem to vmap dataset

        # _, exe_path_BSOPA, _ = self.execute_mpc(
        #     self.make_postmean_func_const_key(),
        #     # self.make_postmean_func(),
        #     curr_obs,
        #     jax.tree_util.tree_map(lambda x: x[0], batch_train_state),
        #     (train_data.X, train_data.y),
        #     batch_key[0],
        #     self.env_params.horizon,
        #     self.agent_config.ACTIONS_PER_PLAN)
        # exe_path_BSOPA = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), exe_path_BSOPA)

        # add in some test values
        key, _key = jrandom.split(key)
        x_test = jnp.concatenate((curr_obs, self.env.action_space(self.env_params).sample(_key)))

        # now optimise the dynamics model with the x_test
        # take the exe_path_list that has been found with different posterior samples using iCEM
        # x_data and y_data are what ever you have currently
        key, _key = jrandom.split(key)
        x_next, acq_val = self._optimise(train_state, train_data, self.make_postmean_func(), exe_path_BSOPA, x_test, _key)

        assert jnp.allclose(curr_obs, x_next[:self.obs_dim]), "For rollout cases, we can only give queries which are from the current state"
        # TODO can we make this jittable?

        return x_next, exe_path_BSOPA, curr_obs, train_state, acq_val, key
