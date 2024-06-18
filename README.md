# Jax Distributed Demo

This code demonstrates using [JAX](https://jax.readthedocs.io/en/latest/index.html) in an idiomatic distributed fashion at [NERSC](https://docs.nersc.gov/).

It starts a job running over several nodes, with one process per GPU per node.
Each process generates some local data, small enough to fit in their GPU, then the data is aggregated into a larger sharded array.
We run a computation over that sharded data and return.

The code is mainly inspired by [this tutorial](https://github.com/ASKabalan/Jax-multihost/tree/main).

## Usage

To run, start the `container.slurm` script in the same folder as the source (you will need to change the `account` name):

```sh
sbatch container.slurm
```

Notice that the script loads an [NVIDIA JAX container](https://github.com/NVIDIA/JAX-Toolbox) with GPU enabled instead of using modules.
This ensures that we have a working installation of JAX that is compatible with distributed computation (something that can be complicated to set up from scratch).

See the [NERSC Shifter documentation](https://docs.nersc.gov/development/containers/shifter/how-to-use/#using-shifter-at-nersc) for further information on container usage (such as using the `mpich` module to enable MPI use).

## Files

* `container.slurm` contains our Slurm script,
* `distributed.py` contains the demo Python script ([make_array_from_single_device_arrays](https://jax.readthedocs.io/en/latest/_autosummary/jax.make_array_from_single_device_arrays.html#jax.make_array_from_single_device_arrays) based),
* `distributed_local_to_global.py` contains an alernative implementation ([multihost_utils.host_local_array_to_global_array](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.multihost_utils.host_local_array_to_global_array.html) based),
* `output.out` contains a typical output.
