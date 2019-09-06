from copy import deepcopy
import functools
import random

try:
    import numpy
    import mxnet
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

__all__ = ['run_binary_op_benchmark', 'run_unary_op_benchmark', 'run_op_frameworks_benchmark',
           'run_creation_op_benchmark']


def _run_simple_op_benchmark(num_input, op, config, mode='forward', warmup=10, runs=25):
    if not op.get_forward_func():
        return None, config
    backend = backend_switcher[op.get_backend()]
    func = op.get_forward_func()
    if num_input:
        config_ = deepcopy(config)
        tensor_config = {'shape': config_.pop('shape'), 'dtype': config_.pop('dtype')} if num_input else None
        func = functools.partial(func, **config_)
        if backend == 'numpy':
            def benchmark_func(inputs):
                result = func(*inputs)
                return result
            input_func = functools.partial(prepare_numpy_inputs, num_input, tensor_config)
            forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
            return forward_time, config
        elif backend == 'mxnet.numpy':
            if mode == 'forward':
                def benchmark_func(inputs):
                    result = func(*inputs)
                    return result
                input_func = functools.partial(prepare_mxnet_inputs, num_input, tensor_config, False)
                forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
                return forward_time, config
            else:
                input_func = functools.partial(prepare_mxnet_inputs, num_input, tensor_config, True)

                def run_graph(inputs):
                    with mxnet.autograd.record():
                        result = func(*inputs)
                    result.backward()
                    return result
                both_time, _ = get_time_metric(run_graph, input_func, warmup, runs)
                return both_time, config
        elif backend == 'jax.numpy':
            input_func = functools.partial(prepare_jax_inputs, num_input, tensor_config)
            if mode == 'forward':
                jit_func = jax.jit(func, list(range(num_input)))

                def benchmark_func(inputs):
                    result = jit_func(*inputs)
                    try:
                        result.block_until_ready()
                    except Exception:
                        pass
                    return result

                forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
                return forward_time, config
            else:
                def grad_func(*args):
                    return jax.numpy.sum(func(*args))
                jit_func = jax.jit(jax.grad(grad_func, list(range(num_input))),
                                   list(range(num_input)))

                def benchmark_func(inputs):
                    result = jit_func(*inputs)
                    try:
                        result.block_until_ready()
                    except Exception:
                        pass
                    return result
                both_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
                return both_time, config
        elif backend == 'chainerx':
            device = chainerx.get_default_device()
            if mode == 'forward':
                input_func = functools.partial(prepare_chainerx_inputs, num_input, tensor_config, False)

                def benchmark_func(inputs):
                    res = func(*inputs)
                    device.synchronize()
                    return res
                forward_time, _ = get_time_metric(benchmark_func, input_func, warmup, runs)
                return forward_time, config
            else:
                input_func = functools.partial(prepare_chainerx_inputs, num_input, tensor_config, True)

                def run_graph(inputs):
                    result = func(*inputs)
                    result.grad = chainerx.ones_like(result)
                    result.backward()
                    device.synchronize()
                    return result
                both_time, _ = get_time_metric(run_graph, input_func, warmup, runs)
                return both_time, config
    else:
        func = functools.partial(func, **config)
        forward_time, _ = get_time_metric(func, None, warmup, runs)
        return forward_time, config


def run_creation_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    return _run_simple_op_benchmark(0, op, config, mode, warmup, runs)


def run_unary_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    return _run_simple_op_benchmark(1, op, config, mode, warmup, runs)


def run_binary_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    return _run_simple_op_benchmark(2, op, config, mode, warmup, runs)


def run_op_frameworks_benchmark(opc, config_func, benchmark_func, backends,
                                mode='forward', times=6, warmup=10, runs=25):
    if not isinstance(backends, list):
        raise Warning("Argument 'backends' must be a list")
    if isinstance(config_func(), list):
        result_list = []
        config_list = config_func()
        for config in config_list:
            np_seed = random.randint(0, 10000)
            result = {}
            for backend in backends:
                backend_ = backend_switcher[backend]
                try:
                    numpy.random.seed(np_seed)
                    result[backend_] = benchmark_func(opc(backend_), config, mode, warmup, runs)[0] * 1000
                except Exception:
                    result[backend_] = None
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
                backend_ = backend_switcher[backend]
                try:
                    numpy.random.seed(np_seed)
                    result[backend_] = benchmark_func(opc(backend_), config, mode, warmup, runs)[0] * 1000
                except Exception:
                    result[backend_] = None
            result['config'] = config
            result_list.append(result)
        return result_list
