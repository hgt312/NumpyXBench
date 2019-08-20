from copy import deepcopy
import functools
from random import randint

try:
    import numpy
    import mxnet
    import torch
    import chainerx
    import jax
except Exception:
    pass

from .common import backend_switcher
from .metrics import *
from .mxnet_util import *
from .numpy_util import *
from .jax_util import *
from .chainerx_util import *
from .torch_util import *

__all__ = ['run_binary_op_benchmark', 'run_unary_op_benchmark', 'run_op_frameworks_benchmark',
           'run_creation_op_benchmark', 'run_withaxis_unary_benchmark']


def _run_xnary_op_benchmark(num_input, op, config, mode='forward', warmup=10, runs=25):
    backend = backend_switcher[op.get_backend()]
    func = op.get_forward_func()
    if backend == 'numpy':
        func = functools.partial(func, *prepare_numpy_inputs(num_input, config))
        forward_time, _ = get_time_metric(func, warmup, runs)
        if mode != 'forward':
            raise Warning("Numpy has no backward")
        return forward_time, config
    elif backend == 'mxnet.numpy':
        if mode == 'forward':
            func = functools.partial(func, *prepare_mxnet_inputs(num_input, config, False))
            forward_time, _ = get_time_metric(func, warmup, runs)
            return forward_time, config
        else:
            def run_graph():
                with mxnet.autograd.record():
                    result = func(*prepare_mxnet_inputs(num_input, config, True))
                result.backward()
            both_time = get_time_metric(run_graph, warmup, runs)
            return both_time, config
    elif backend == 'jax.numpy':
        jit_func = jax.jit(func, list(range(num_input)))
        benchmark_func = functools.partial(jit_func, *prepare_jax_inputs(num_input, config))
        forward_time, _ = get_time_metric(benchmark_func, warmup, runs)
        if mode == 'forward':
            return forward_time, config
        else:
            def grad_func(*args):
                return jax.numpy.sum(func(*args))
            jit_func = jax.jit(jax.grad(grad_func, list(range(num_input))),
                               list(range(num_input)))
            benchmark_func = functools.partial(jit_func, *prepare_jax_inputs(num_input, config))
            backward_time, _ = get_time_metric(benchmark_func, warmup, runs)
            return forward_time + backward_time, config
    elif backend == 'chainerx':
        if mode == 'forward':
            func = functools.partial(func, *prepare_chainerx_inputs(num_input, config, False))
            forward_time, _ = get_time_metric(func, warmup, runs)
            return forward_time, config
        else:
            def run_graph():
                result = func(*prepare_chainerx_inputs(num_input, config, True))
                result.grad = chainerx.ones_like(result)
                result.backward()
            both_time = get_time_metric(run_graph, warmup, runs)
            return both_time, config


def run_creation_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    backend = backend_switcher[op.get_backend()]
    func = op.get_forward_func()
    # TODO(hgt312): jax jit
    if backend in ['numpy', 'mxnet.numpy', 'chainerx', 'jax.numpy']:
        func = functools.partial(func, **config)
        forward_time, _ = get_time_metric(func, warmup, runs)
        return forward_time, config


def run_binary_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    return _run_xnary_op_benchmark(2, op, config, mode, warmup, runs)


def run_unary_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    return _run_xnary_op_benchmark(1, op, config, mode, warmup, runs)


def run_withaxis_unary_benchmark(op, config, mode='forward', warmup=10, runs=25):
    backend = backend_switcher[op.get_backend()]
    func = op.get_forward_func()
    config_ = deepcopy(config)
    axis = config_.pop('axis')
    if backend == 'numpy':
        func = functools.partial(func, *prepare_numpy_inputs(1, config_), axis=axis)
        forward_time, _ = get_time_metric(func, warmup, runs)
        if mode != 'forward':
            raise Warning("Numpy has no backward")
        return forward_time, config
    elif backend == 'mxnet.numpy':
        if mode == 'forward':
            func = functools.partial(func, *prepare_mxnet_inputs(1, config_, False), axis=axis)
            forward_time, _ = get_time_metric(func, warmup, runs)
            return forward_time, config
        else:
            def run_graph():
                with mxnet.autograd.record():
                    result = func(*prepare_mxnet_inputs(2, config_, True), axis=axis)
                result.backward()

            both_time = get_time_metric(run_graph, warmup, runs)
            return both_time, config
    elif backend == 'jax.numpy':
        jit_func = jax.jit(func, (0, 1))
        benchmark_func = functools.partial(jit_func, *prepare_jax_inputs(1, config), axis)
        forward_time, _ = get_time_metric(benchmark_func, warmup, runs)
        if mode == 'forward':
            return forward_time, config
        else:
            def grad_func(*args):
                return jax.numpy.sum(func(*args))

            jit_func = jax.jit(jax.grad(grad_func), (0, 1))
            benchmark_func = functools.partial(jit_func, *prepare_jax_inputs(1, config), axis)
            backward_time, _ = get_time_metric(benchmark_func, warmup, runs)
            return forward_time + backward_time, config
    elif backend == 'chainerx':
        if mode == 'forward':
            func = functools.partial(func, *prepare_chainerx_inputs(1, config_, False), axis=axis)
            forward_time, _ = get_time_metric(func, warmup, runs)
            return forward_time, config
        else:
            def run_graph():
                result = func(*prepare_chainerx_inputs(1, config_, True), axis=axis)
                result.grad = chainerx.ones_like(result)
                result.backward()

            both_time = get_time_metric(run_graph, warmup, runs)
            return both_time, config


def run_op_frameworks_benchmark(opc, config_func, benchmark_func, backends,
                                mode='forward', warmup=10, runs=25, seed=None):
    if not isinstance(backends, list):
        raise Warning("Argument 'backends' must be a list")
    if not seed:
        seed = randint(0, 10000)
    config = config_func()
    result = {}
    for backend in backends:
        try:
            numpy.random.seed(seed)
            result[backend] = benchmark_func(opc(backend), config, mode, warmup, runs)[0]
        except Exception:
            result[backend] = None
    result['config'] = config
    return result
