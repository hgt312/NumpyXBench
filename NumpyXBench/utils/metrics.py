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


def get_time_metric(func, warmup=10, runs=25):
    timeit.timeit(func, number=warmup)
    return timeit.timeit(func, number=runs)
