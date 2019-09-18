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
           'run_creation_op_benchmark', 'run_binary_broadcast_op_benchmark']


def _run_simple_op_benchmark(num_input, op, config, mode='forward', warmup=10, runs=25):
    if not op.get_forward_func():
        return (None, None), config
    backend = backend_switcher[op.get_backend()]
    func = op.get_forward_func()
    if num_input:
        config_ = deepcopy(config)
        tensor_config = {'shape': config_.pop('shape'), 'dtype': config_.pop('dtype')} if num_input else None
        func = functools.partial(func, **config_)
        if backend == 'numpy':
            if mode == 'forward':
                def benchmark_func(inputs):
                    result = func(*inputs)
                    return result
                input_func = functools.partial(prepare_numpy_inputs, num_input, tensor_config)
                forward_time, forward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
                return (forward_time, forward_std), config
            else:
                return (None, None), config
        elif backend == 'mxnet.numpy':
            if mode == 'forward':
                def benchmark_func(inputs):
                    result = func(*inputs)
                    return result
                input_func = functools.partial(prepare_mxnet_inputs, num_input, tensor_config, False)
                forward_time, forward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
                return (forward_time, forward_std), config
            else:
                def input_func():
                    inputs = prepare_mxnet_inputs(num_input, tensor_config, True)
                    with mxnet.autograd.record():
                        result = func(*inputs)
                    return result

                def benchmark_func(result):
                    result.backward()
                backward_time, backward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
                return (backward_time, backward_std), config
        elif backend == 'jax.numpy':
            input_func = functools.partial(prepare_jax_inputs, num_input, tensor_config)
            if mode == 'forward':
                jit_func = jax.jit(func)

                def benchmark_func(inputs):
                    result = jit_func(*inputs)
                    try:
                        result.block_until_ready()
                    except Exception:
                        pass
                    return result

                forward_time, forward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
                return (forward_time, forward_std), config
            else:
                def grad_func(*args):
                    return jax.numpy.sum(func(*args))
                jit_func = jax.jit(jax.grad(grad_func, list(range(num_input))))

                def benchmark_func(inputs):
                    result = jit_func(*inputs)
                    try:
                        result.block_until_ready()
                    except Exception:
                        pass
                    return result
                backward_time, backward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
                return (backward_time, backward_std), config
        elif backend == 'chainerx':
            device = chainerx.get_default_device()
            if mode == 'forward':
                input_func = functools.partial(prepare_chainerx_inputs, num_input, tensor_config, False)

                def benchmark_func(inputs):
                    res = func(*inputs)
                    device.synchronize()
                    return res
                forward_time, forward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
                return (forward_time, forward_std), config
            else:
                def input_func():
                    inputs = prepare_chainerx_inputs(num_input, tensor_config, True)
                    result = func(*inputs)
                    result.grad = chainerx.ones_like(result)
                    return result

                def benchmark_func(result):
                    result.backward()
                    device.synchronize()
                backward_time, backward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
                return (backward_time, backward_std), config
    else:
        if mode == 'forward':
            func = functools.partial(func, **config)
            forward_time, forward_std = get_time_metric(func, None, warmup, runs)
            return (forward_time, forward_std), config
        else:
            return (None, None), config


def run_creation_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    return _run_simple_op_benchmark(0, op, config, mode, warmup, runs)


def run_unary_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    return _run_simple_op_benchmark(1, op, config, mode, warmup, runs)


def run_binary_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    return _run_simple_op_benchmark(2, op, config, mode, warmup, runs)


def run_binary_broadcast_op_benchmark(op, config, mode='forward', warmup=10, runs=25):
    if not op.get_forward_func():
        return (None, None), config
    backend = backend_switcher[op.get_backend()]
    func = op.get_forward_func()
    config1 = {"shape": config["shape1"], "dtype": config["dtype"]}
    config2 = {"shape": config["shape2"], "dtype": config["dtype"]}
    if backend == 'numpy':
        if mode == 'forward':
            def benchmark_func(inputs):
                result = func(*inputs)
                return result

            def input_func():
                return prepare_numpy_inputs(1, config1) + prepare_numpy_inputs(1, config2)
            forward_time, forward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
            return (forward_time, forward_std), config
        else:
            return (None, None), config
    elif backend == 'mxnet.numpy':
        if mode == 'forward':
            def benchmark_func(inputs):
                result = func(*inputs)
                return result

            def input_func():
                return prepare_mxnet_inputs(1, config1, False) + prepare_mxnet_inputs(1, config2, False)
            forward_time, forward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
            return (forward_time, forward_std), config
        else:
            def input_func():
                inputs = prepare_mxnet_inputs(1, config1, True) + prepare_mxnet_inputs(1, config2, True)
                with mxnet.autograd.record():
                    result = func(*inputs)
                return result

            def benchmark_func(result):
                result.backward()
                return result

            backward_time, backward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
            return (backward_time, backward_std), config
    elif backend == 'jax.numpy':
        def input_func():
            return prepare_jax_inputs(1, config1) + prepare_jax_inputs(1, config2)
        if mode == 'forward':
            jit_func = jax.jit(func)

            def benchmark_func(inputs):
                result = jit_func(*inputs)
                try:
                    result.block_until_ready()
                except Exception:
                    pass
                return result

            forward_time, forward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
            return (forward_time, forward_std), config
        else:
            def grad_func(*args):
                return jax.numpy.sum(func(*args))

            jit_func = jax.jit(jax.grad(grad_func, [0, 1]))

            def benchmark_func(inputs):
                result = jit_func(*inputs)
                try:
                    result.block_until_ready()
                except Exception:
                    pass
                return result

            backward_time, backward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
            return (backward_time, backward_std), config
    elif backend == 'chainerx':
        device = chainerx.get_default_device()
        if mode == 'forward':
            def input_func():
                return prepare_chainerx_inputs(1, config1, False) + prepare_chainerx_inputs(1, config2, False)

            def benchmark_func(inputs):
                res = func(*inputs)
                device.synchronize()
                return res

            forward_time, forward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
            return (forward_time, forward_std), config
        else:
            def input_func():
                inputs = prepare_chainerx_inputs(1, config1, True) + prepare_chainerx_inputs(1, config2, True)
                result = func(*inputs)
                result.grad = chainerx.ones_like(result)
                return result

            def benchmark_func(result):
                result.backward()
                device.synchronize()

            backward_time, backward_std = get_time_metric(benchmark_func, input_func, warmup, runs)
            return (backward_time, backward_std), config


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
                    result[backend_] = benchmark_func(opc(backend_), config, mode, warmup, runs)[0]
                except Exception:
                    result[backend_] = (None, None)
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
                    result[backend_] = benchmark_func(opc(backend_), config, mode, warmup, runs)[0]
                except Exception:
                    result[backend_] = (None, None)
            result['config'] = config
            result_list.append(result)
        return result_list
