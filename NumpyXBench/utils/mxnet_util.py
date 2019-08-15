try:
    import mxnet
except Exception:
    pass

from .numpy_util import prepare_numpy_inputs

__all__ = ['prepare_mxnet_inputs']


def prepare_mxnet_inputs(num_input, config, grad=False, device=None):
    dtype = config['dtype']
    inputs = prepare_numpy_inputs(num_input, config)
    inputs = [mxnet.numpy.array(i, dtype=dtype) for i in inputs]
    if grad:
        for i in inputs:
            i.attach_grad()
    return inputs
