try:
    import jax
except Exception:
    pass

from .numpy_util import prepare_numpy_inputs

__all__ = ['prepare_jax_inputs']


def prepare_jax_inputs(num_input, config, grad=False, device=None):
    dtype = config['dtype']
    inputs = prepare_numpy_inputs(num_input, config)
    inputs = [jax.numpy.array(i, dtype=dtype) for i in inputs]
    for i in inputs:
        i.block_until_ready()
    return inputs
