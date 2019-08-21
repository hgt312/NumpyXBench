from functools import partial

from ..configs import *
from .. import operators as ops
from ..utils.benchmarks import *
from ..utils.common import *

from .helpers import get_dtypes

__all__ = ['get_add_blob', 'get_subtract_blob', 'get_multiply_blob', 'get_divide_blob', 'get_mod_blob']


# elemwise binary ops
def get_add_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Add,
            config_func,
            run_binary_op_benchmark), 'add'


def get_subtract_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Subtract,
            config_func,
            run_binary_op_benchmark), 'subtract'


def get_multiply_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Multiply,
            config_func,
            run_binary_op_benchmark), 'multiply'


def get_divide_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Divide,
            config_func,
            run_binary_op_benchmark), 'divide'


def get_mod_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Mod,
            config_func,
            run_binary_op_benchmark), 'mod'
