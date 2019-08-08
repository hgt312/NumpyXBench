import numpy


def prepare_numpy_inputs(num_input, config):
    input_shape = config['shape']
    dtype = config['dtype']
    inputs = []
    for _ in range(num_input):
        inputs.append(numpy.random.normal(size=input_shape).astype(dtype))
    return inputs
