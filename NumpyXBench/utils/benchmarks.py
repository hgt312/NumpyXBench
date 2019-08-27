from copy import deepcopy
import functools
import random

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
           'run_creation_op_benchmark', 'run_withaxis_unary_benchmark', 'run_shape_like_op_benchmark']


def _run_xnary_op_benchmark(num_input, op, config, mode='forward', warmup=10, runs=25):
    backend = backend_switcher[op.get_backend()]
    func = op.get_forward_func()
    if backend == 'numpy':
        def benchmark_func(inputs):
            result = func(*inputs)
            return result
        input_func = functools.partial(prepare_numpy_inputs, num_input, config)
        forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
        if mode != 'forward':
            raise Warning("Numpy has no backward")
        return forward_time, config
    elif backend == 'mxnet.numpy':
        if mode == 'forward':
            def benchmark_func(inputs):
                result = func(*inputs)
                return result
            input_func = functools.partial(prepare_mxnet_inputs, num_input, config, False)
            forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
            return forward_time, config
        else:
            input_func = functools.partial(prepare_mxnet_inputs, num_input, config, True)

            def run_graph(inputs):
                with mxnet.autograd.record():
                    result = func(*inputs)
                result.backward()
                return result
            both_time = get_time_metric(run_graph, input_func, warmup, runs)
            return both_time, config
    elif backend == 'jax.numpy':
        input_func = functools.partial(prepare_jax_inputs, num_input, config)
        jit_func = jax.jit(func, list(range(num_input)))

        def benchmark_func(inputs):
            result = jit_func(*inputs)
            result.block_until_ready()
            return result
        forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
        if mode == 'forward':
            return forward_time, config
        else:
            def grad_func(*args):
                return jax.numpy.sum(func(*args))
            jit_func = jax.jit(jax.grad(grad_func, list(range(num_input))),
                               list(range(num_input)))

            def benchmark_func(inputs):
                result = jit_func(*inputs)
                result.block_until_ready()
                return result
            backward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
            return forward_time + backward_time, config
    elif backend == 'chainerx':
        device = chainerx.get_default_device()
        if mode == 'forward':
            input_func = functools.partial(prepare_chainerx_inputs, num_input, config, False)

            def benchmark_func(inputs):
                res = func(*inputs)
                device.synchronize()
                return res
            forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
            return forward_time, config
        else:
            input_func = functools.partial(prepare_chainerx_inputs, num_input, config, True)

            def run_graph(inputs):
                result = func(*inputs)
                result.grad = chainerx.ones_like(result)
                result.backward()
                device.synchronize()
                return result
            both_time = get_time_metric(run_graph, input_func, warmup, runs)
            return both_time, config


def run_creation_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    backend = backend_switcher[op.get_backend()]
    func = op.get_forward_func()
    # TODO(hgt312): jax jit
    if backend in ['numpy', 'mxnet.numpy', 'chainerx', 'jax.numpy']:
        func = functools.partial(func, **config)
        forward_time, _ = get_time_metric(func, None, warmup, runs)
        return forward_time, config


def run_shape_like_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    backend = backend_switcher[op.get_backend()]
    func = op.get_forward_func()
    if backend == 'numpy':
        def benchmark_func(inputs):
            result = func(*inputs)
            return result
        input_func = functools.partial(prepare_numpy_inputs, 1, config)
        forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
        return forward_time, config
    elif backend == 'mxnet.numpy':
        def benchmark_func(inputs):
            result = func(*inputs)
            return result
        input_func = functools.partial(prepare_mxnet_inputs, 1, config, False)
        forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
        return forward_time, config
    elif backend == 'jax.numpy':
        input_func = functools.partial(prepare_jax_inputs, 1, config)
        jit_func = jax.jit(func, [0])

        def benchmark_func(inputs):
            result = jit_func(*inputs)
            return result
        forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
        return forward_time, config
    elif backend == 'chainerx':
        device = chainerx.get_default_device()
        input_func = functools.partial(prepare_chainerx_inputs, 1, config, False)

        def benchmark_func(inputs):
            res = func(*inputs)
            device.synchronize()
            return res
        forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
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
        def benchmark_func(inputs):
            result = func(*inputs, axis=axis)
            return result
        input_func = functools.partial(prepare_numpy_inputs, 1, config_)
        forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
        if mode != 'forward':
            raise Warning("Numpy has no backward")
        return forward_time, config
    elif backend == 'mxnet.numpy':
        if mode == 'forward':
            def benchmark_func(inputs):
                result = func(*inputs, axis=axis)
                return result
            input_func = functools.partial(prepare_mxnet_inputs, 1, config_, False)
            forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
            return forward_time, config
        else:
            input_func = functools.partial(prepare_mxnet_inputs, 1, config_, True)

            def run_graph(inputs):
                with mxnet.autograd.record():
                    result = func(*inputs, axis=axis)
                result.backward()
                return result
            both_time = get_time_metric(run_graph, input_func, warmup, runs)
            return both_time, config
    elif backend == 'jax.numpy':
        input_func = functools.partial(prepare_jax_inputs, 1, config_)
        jit_func = jax.jit(func, (0, 1))

        def benchmark_func(inputs):
            result = jit_func(*inputs, axis)
            result.block_until_ready()
            return result
        forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
        if mode == 'forward':
            return forward_time, config
        else:
            def grad_func(*args):
                return jax.numpy.sum(func(*args))
            jit_func = jax.jit(jax.grad(grad_func), (0, 1))

            def benchmark_func(inputs):
                result = jit_func(*inputs, axis)
                result.block_until_ready()
                return result
            backward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
            return forward_time + backward_time, config
    elif backend == 'chainerx':
        device = chainerx.get_default_device()
        if mode == 'forward':
            input_func = functools.partial(prepare_chainerx_inputs, 1, config_, False)

            def benchmark_func(inputs):
                res = func(*inputs, axis=axis)
                device.synchronize()
                return res
            forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
            return forward_time, config
        else:
            input_func = functools.partial(prepare_chainerx_inputs, 1, config_, True)

            def run_graph(inputs):
                result = func(*inputs, axis=axis)
                result.grad = chainerx.ones_like(result)
                result.backward()
                device.synchronize()
                return result
            both_time = get_time_metric(run_graph, input_func, warmup, runs)
            return both_time, config


def run_op_frameworks_benchmark(opc, config_func, benchmark_func, backends,
                                mode='forward', is_random=True, times=6, warmup=10, runs=25):
    if not isinstance(backends, list):
        raise Warning("Argument 'backends' must be a list")
    if (not is_random) and isinstance(config_func(), list):
        result_list = []
        config_list = config_func()
        for config in config_list:
            np_seed = random.randint(0, 10000)
            result = {}
            for backend in backends:
                try:
                    numpy.random.seed(np_seed)
                    result[backend] = benchmark_func(opc(backend), config, mode, warmup, runs)[0]
                except Exception:
                    result[backend] = float('inf')
            result['config'] = config
            result_list.append(result)
        return result_list
    else:
        result_list = []
        for t in range(times):
            config = config_func()
            np_seed = random.randint(0, 10000)
            result = {}
            for backend in backends:
                try:
                    numpy.random.seed(np_seed)
                    result[backend] = benchmark_func(opc(backend), config, mode, warmup, runs)[0]
                except Exception:
                    result[backend] = float('inf')
            result['config'] = config
            result_list.append(result)
        return result_list
