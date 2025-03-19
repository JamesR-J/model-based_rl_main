import jax.numpy as jnp

a = jnp.arange(10).reshape(2, 1, 5) + 10

idx = jnp.argmax(a)

multi_index = jnp.unravel_index(idx, a.shape)

maximal = a[multi_index]

print(a)
print(idx)
print(multi_index)
print(maximal)