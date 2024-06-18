import jax
from jax import random
import jax.numpy as jnp
from jax.experimental import multihost_utils

# Initializes distributed JAX
jax.distributed.initialize()

# Displays the devices accesible
verbose = (jax.process_index() == 0)
print(f"[{jax.process_index()}]: local devices: {jax.local_devices()}")
if verbose: print(f"Global devices: {jax.devices()}")

# Defines the size of the matrix and vector
local_size = 40000 # fits a single A100 GPU

# Creates local data
key = random.PRNGKey(0)
key_matrix, key_vector = random.split(key)
local_matrix = random.normal(key_matrix, (local_size, local_size))
local_vector = random.normal(key_vector, (local_size,))
if verbose: jax.debug.visualize_array_sharding(local_matrix) # one small matrix, local to the local device

# Defines our sharding strategy
# see: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
mesh = jax.sharding.Mesh(jax.devices(), 'devices')
matrix_partition_spec = jax.sharding.PartitionSpec(None, 'devices')
vector_partition_spec = jax.sharding.PartitionSpec('devices')

# Consolidates local data into sharded data
# see: https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.multihost_utils.host_local_array_to_global_array.html
sharded_matrix = multihost_utils.host_local_array_to_global_array(local_matrix, mesh, matrix_partition_spec)
sharded_vector = multihost_utils.host_local_array_to_global_array(local_vector, mesh, vector_partition_spec)
if verbose: 
    print(f"sharded_matrix.shape: {sharded_matrix.shape}") # (40000, 320000)
    print(f"sharded_vector.shape: {sharded_vector.shape}") # (320000,)
    jax.debug.visualize_array_sharding(sharded_matrix) # one matrix, too big for local memory, distributed over all devices along its second axis

# Performs computation with sharded data
# NOTE: this should be jit-compiled
sharded_result = jnp.dot(sharded_matrix, sharded_vector)
if verbose: 
    print(f"sharded_result.shape: {sharded_result.shape}") # (40000,)
    jax.debug.visualize_array_sharding(sharded_result) # one vector distributed over all devices

# Finished
print(f"[{jax.process_index()}]: Done.")
