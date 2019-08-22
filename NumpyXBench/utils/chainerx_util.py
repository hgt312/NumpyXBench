try:
    import chainerx
except Exception:
    pass

from .numpy_util import prepare_numpy_inputs

__all__ = ['prepare_chainerx_inputs']


def prepare_chainerx_inputs(num_input, config, grad=False):
    device = chainerx.get_default_device()
    dtype = config['dtype']
    inputs = prepare_numpy_inputs(num_input, config)
    inputs = [chainerx.array(i, dtype=dtype) for i in inputs]
    if grad:
        for i in inputs:
            i.require_grad()
    device.synchronize()
    return inputs
