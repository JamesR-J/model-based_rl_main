from project_name.dynamics_models import MOSVGPGPJax
import jax.numpy as jnp
import jax.random as jrandom
from project_name.envs import GymnaxPendulum
from ml_collections import config_dict
import gpjax
import jax.scipy as jsp
import jax


jax.config.update("jax_enable_x64", True)


# TODO generalise this to arbitrary num_datapoints, num_inputs, num_outputs


env = GymnaxPendulum()
env_params = env.default_params
key = jrandom.key(42)

output_dim = 2

config = config_dict.ConfigDict()
config.LEARN_REWARD = False

agent_config = config_dict.ConfigDict()
agent_config.NUM_INDUCING_POINTS = 100

mean = gpjax.mean_functions.Constant(jnp.array((0.07455202985890419)))
kernel1 = gpjax.kernels.RBF(active_dims=[0, 1, 2], lengthscale=jnp.array((2.81622296,   9.64035469, 142.60660018)), variance=0.78387795)  # TODO make this more general
kernel2 = gpjax.kernels.RBF(active_dims=[0, 1, 2], lengthscale=jnp.array((0.92813981, 280.24169475,  14.85778016)), variance=0.22877621)  # TODO make this more general
gps = [gpjax.gps.Prior(mean_function=mean, kernel=kernel1), gpjax.gps.Prior(mean_function=mean, kernel=kernel2)]

gp = MOSVGPGPJax(env, env_params, config, agent_config, key)

key, _key = jrandom.split(key)
init_obs, env_state = env.reset(_key)
init_data = gpjax.Dataset(jnp.expand_dims(jnp.concatenate((init_obs, jnp.zeros((1,)))), axis=0),
                                    jnp.expand_dims(init_obs, axis=0))
train_state = gp.create_train_state(init_data.X, init_data.y, _key)
m = jnp.ones((1, 3))
s = jnp.ones((3, 3))

# gp.predict_on_noisy_inputs(m, s, train_state, init_data)

num_data = 120  # 9  # 12
new_data = gpjax.Dataset(jnp.linspace(0, num_data-1, num_data).reshape(-1, 3),
                         jnp.linspace(num_data, num_data+((num_data // 3) * output_dim) - 1, (num_data // 3) * output_dim).reshape(-1, output_dim))
data = gp._adjust_dataset(new_data)

K = []
for gp_idx, prior in enumerate(gps):
    var_posterior = gpjax.variational_families.VariationalGaussian(posterior=prior * gpjax.likelihoods.Gaussian(num_datapoints=new_data.n, obs_stddev=gpjax.parameters.PositiveReal(jnp.array(0.005988507226896687))), inducing_inputs=gp.og_z)
    D_dim = gpjax.Dataset(X=new_data.X, y=jnp.expand_dims(new_data.y[:, gp_idx], axis=-1))
    K.append(var_posterior.posterior.prior.kernel.gram(D_dim.X).A)
K = jnp.array(K)
batched_eye = jnp.expand_dims(jnp.eye(jnp.shape(new_data.X)[0]), axis=0).repeat(output_dim, axis=0)
noise = jnp.repeat(jnp.array(0.005988507226896687), output_dim) ** 2
L = jsp.linalg.cho_factor(K + noise[:, None, None] * batched_eye * var_posterior.posterior.jitter, lower=True)
iK = jsp.linalg.cho_solve(L, batched_eye)
Y_ = jnp.transpose(new_data.y)[:, :, None]
beta = jsp.linalg.cho_solve(L, Y_)[:, :, 0]

q_posterior = gp.variational_posterior_builder(data.n)
mo_iK, mo_beta = gp._calculate_factorisations(q_posterior.posterior, data)
mo_beta = mo_beta.flatten().reshape(-1, output_dim).T


def reconstruct_original(matrix, k=2):
    matrix1 = matrix[::k, ::k]
    matrix2 = matrix[1::k, 1::k]
    return jnp.concatenate((jnp.expand_dims(matrix1, axis=0), jnp.expand_dims(matrix2, axis=0)), axis=0)

mo_iK = reconstruct_original(mo_iK)

# print(beta.shape)
# print(beta)
# print(mo_beta.shape)
# print(mo_beta)

assert jnp.allclose(beta, mo_beta), "Beta values are different"

# print(iK.shape)
# print(iK)
# print(mo_iK.shape)
# print(mo_iK)

assert jnp.allclose(iK, mo_iK), "iK values are different"

# TODO want the output to be also of shape 2, data, data




