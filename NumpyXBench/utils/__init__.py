import time

import numpy
import mxnet
import torch

from ..configs import get_random_size_config


def run_binary_op_benchmark(op, config, mode='forward'):
    backend = op.get_backend()
    # print('Backend: {}'.format(backend))
    config = get_random_size_config()
    input_shape = config['shape']
    dtype = config['dtype']
    if backend in ['numpy', 'np']:
        lhs = numpy.random.normal(size=input_shape).astype(dtype)
        rhs = numpy.random.normal(size=input_shape).astype(dtype)
        func = op.get_forward_func()
        start_time = time.time()
        result = func(lhs, rhs)
        end_time = time.time()
        if mode != 'forward':
            raise Warning("Numpy has no backward")
        return end_time - start_time, config
    elif backend in ['mxnet', 'mx']:
        with mxnet.autograd.record():
            lhs = mxnet.numpy.random.normal(size=input_shape, dtype=dtype).astype(dtype)
            rhs = mxnet.numpy.random.normal(size=input_shape, dtype=dtype).astype(dtype)
        func = op.get_forward_func()
        start_time1 = time.time()
        result = func(lhs, rhs)
        end_time1 = time.time()
        forward_time = end_time1 - start_time1
        if mode == 'both':
            start_time2 = time.time()
            result.backward()
            end_time2 = time.time()
            backward_time = end_time2 - start_time2
            return forward_time, backward_time, config
        else:
            return forward_time, config
