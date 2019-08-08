try:
    import mxnet
except ImportError:
    pass


def prepare_mxnet_inputs(num_input, config, grad=False, device=None):
    input_shape = config['shape']
    dtype = config['dtype']
    inputs = []
    for _ in range(num_input):
        inputs.append(mxnet.numpy.random.normal(size=input_shape, dtype=dtype))
    if grad:
        for i in range(num_input):
            inputs[i].attach_grad()
    return inputs
