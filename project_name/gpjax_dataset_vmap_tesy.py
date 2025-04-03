import jax
import jax.numpy as jnp
import jax.random as jrandom
import gpjax


jax.config.update("jax_enable_x64", True)


train_data = gpjax.Dataset(jnp.zeros((1,2)), jnp.zeros((1, 1)))


def rando_func(train_data, key):
    return train_data, key

key = jrandom.key(42)
batch_key = jrandom.split(key, 4)
end_train_data, end_key = jax.vmap(rando_func, in_axes=(None, 0))(train_data, batch_key)