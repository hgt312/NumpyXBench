import functools
import timeit

timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        res = {stmt}
    _t1 = _timer()
    return _t1 - _t0, res
"""


def get_time_metric(benchmark_func, input_func=None, warmup=10, runs=25):
    """
    :param benchmark_func:
    :param input_func: tensors input part, a function without input
    :param warmup:
    :param runs:
    :return:
    """
    results = []
    for _ in range(warmup):
        try:
            if input_func:
                benchmark_func_ = functools.partial(benchmark_func, input_func())
                timeit.timeit(benchmark_func_, number=1)
            else:
                timeit.timeit(benchmark_func, number=1)
        except Exception:
            return None, None
    for _ in range(runs):
        if input_func:
            benchmark_func_ = functools.partial(benchmark_func, input_func())
            result = timeit.timeit(benchmark_func_, number=1)
        else:
            result = timeit.timeit(benchmark_func, number=1)
        results.append(result[0])
    return sum(results) / runs, result[1]
