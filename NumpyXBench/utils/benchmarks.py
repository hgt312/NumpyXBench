import functools

import numpy
import mxnet
import torch

from .metrics import *

# __all__ = ['run_binary_op_benchmark', 'run_op_frameworks_benchmark']

torch_type_switch = {
    'float32': torch.float32,
    'float64': torch.float64,
}


def _prepare_numpy_inputs(num_input, config):
    input_shape = config['shape']
    dtype = config['dtype']
    inputs = []
    for _ in range(num_input):
        inputs.append(numpy.random.normal(size=input_shape).astype(dtype))
    return inputs


def _prepare_mxnet_inputs(num_input, config, grad=False, device=None):
    input_shape = config['shape']
    dtype = config['dtype']
    inputs = []
    for _ in range(num_input):
        inputs.append(mxnet.numpy.random.normal(size=input_shape, dtype=dtype))
    if grad:
        for i in range(num_input):
            inputs[i].attach_grad()
    return inputs


def _prepare_torch_inputs(num_input, config, grad=False, device=None):
    input_shape = config['shape']
    dtype = config['dtype']
    inputs = []
    for _ in range(num_input):
        inputs.append(torch.randn(*input_shape,
                                  dtype=torch_type_switch[dtype],
                                  requires_grad=grad))
    return inputs


def run_binary_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    backend = op.get_backend()
    # print('Backend: {}'.format(backend))
    if backend in ['numpy', 'np']:
        func = op.get_forward_func()
        func = functools.partial(func, *_prepare_numpy_inputs(2, config))
        forward_time, _ = get_time_metric(func)
        if mode != 'forward':
            raise Warning("Numpy has no backward")
        return forward_time, config
    elif backend in ['mxnet', 'mx']:
        func = op.get_forward_func()
        if mode == 'forward':
            func = functools.partial(func, *_prepare_mxnet_inputs(2, config, False))
            forward_time, _ = get_time_metric(func, warmup, runs)
            return forward_time, config
        else:
            def run_graph():
                with mxnet.autograd.record():
                    result = func(*_prepare_mxnet_inputs(2, config, True))
                result.backward()
            both_time = get_time_metric(run_graph, warmup, runs)
            return both_time, config


def run_op_frameworks_benchmark(opc, config_func, benchmark_func, backends, mode='forward', warmup=10, runs=25):
    if not isinstance(backends, list):
        raise Warning("Argument 'backends' must be a list")
    config = config_func()
    result = {'config': config}
    for backend in backends:
        op = opc(backend)
        result[backend] = benchmark_func(op, config, mode, warmup, runs)[0]
    return result
