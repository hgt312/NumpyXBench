try:
    import mxnet
except ImportError:
    pass


def prepare_mxnet_inputs(num_input, config, grad=False, device=None):
    input_shape = config['shape']
    dtype = config['dtype']
    inputs = [mxnet.numpy.random.normal(size=input_shape, dtype=dtype) for _ in range(num_input)]

    if grad:
        for i in inputs:
            i.attach_grad()
    return inputs
