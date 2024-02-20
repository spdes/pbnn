"""
Generate independent random seeds.
"""
import jax
import numpy as np

key = jax.random.PRNGKey(666)
key_jax, key_np = jax.random.split(key)

keys_jax = jax.random.split(key_jax, num=1000)
keys_np = jax.random.split(key_np, num=1000)

np.save('./keys_jax', np.asarray(keys_jax))
np.save('./keys_np', np.asarray(keys_np)[:, 0])
