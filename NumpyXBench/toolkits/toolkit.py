from functools import partial

from .helpers import get_dtypes


class Toolkit(object):
    def __init__(self, has_backward=True, forward_dtypes='AllTypes', backward_dtypes='RealTypes',
                 name=None, operator_cls=None, random_config_func=None, determined_config_func=None,
                 benchmark_func=None):
        self.has_backward = has_backward
        self.forward_dtypes = forward_dtypes
        self.backward_dtypes = backward_dtypes
        self.name = name
        self.operator_cls = operator_cls
        self.random_config_func = random_config_func
        self.determined_config_func = determined_config_func
        self.benchmark_func = benchmark_func
        if not self.has_backward:
            self.backward_dtypes = None

    def get_forward_dtypes(self):
        return self.forward_dtypes

    def get_backward_dtypes(self):
        if self.has_backward:
            return self.backward_dtypes
        else:
            return None

    def get_name(self):
        return self.name

    def get_operator_cls(self):
        return self.operator_cls

    def get_random_config_func(self, dtypes):
        return partial(self.random_config_func, get_dtypes(dtypes))

    def get_determined_config_func(self, dtypes):
        func = self.determined_config_func if self.determined_config_func else self.random_config_func
        return partial(func, get_dtypes(dtypes))

    def get_benchmark_func(self):
        return self.benchmark_func

    def get_tools(self, dtypes, is_random=True):
        if is_random:
            config_func = self.get_random_config_func(dtypes)
        else:
            config_func = self.get_determined_config_func(dtypes)
        return self.operator_cls, config_func, self.benchmark_func
