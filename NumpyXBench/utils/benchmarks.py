import functools

try:
    import numpy
    import mxnet
    import torch
except ImportError:
    pass

from .metrics import *
from .mxnet_util import *
from .numpy_util import *
from .torch_util import *

__all__ = ['run_binary_op_benchmark', 'run_op_frameworks_benchmark']


def run_binary_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    backend = op.get_backend()
    # print('Backend: {}'.format(backend))
    if backend in ['numpy', 'np']:
        func = op.get_forward_func()
        func = functools.partial(func, *prepare_numpy_inputs(2, config))
        forward_time, _ = get_time_metric(func)
        if mode != 'forward':
            raise Warning("Numpy has no backward")
        return forward_time, config
    elif backend in ['mxnet', 'mx']:
        func = op.get_forward_func()
        if mode == 'forward':
            func = functools.partial(func, *prepare_mxnet_inputs(2, config, False))
            forward_time, _ = get_time_metric(func, warmup, runs)
            return forward_time, config
        else:
            def run_graph():
                with mxnet.autograd.record():
                    result = func(*prepare_mxnet_inputs(2, config, True))
                result.backward()

            both_time = get_time_metric(run_graph, warmup, runs)
            return both_time, config


def run_op_frameworks_benchmark(opc, config_func, benchmark_func, backends, mode='forward', warmup=10, runs=25):
    if not isinstance(backends, list):
        raise Warning("Argument 'backends' must be a list")
    config = config_func()
    result = {backend: benchmark_func(opc(backend), config, mode, warmup, runs)[0] for backend in backends}
    result['config'] = config
    return result
