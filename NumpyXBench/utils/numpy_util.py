import numpy


def prepare_numpy_inputs(num_input, config):
    input_shape = config['shape']
    dtype = config['dtype']
    inputs = [numpy.random.normal(size=input_shape).astype(dtype) for _ in range(num_input)]

    return inputs
