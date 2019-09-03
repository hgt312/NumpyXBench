from functools import partial

from ..configs import *
from .. import operators as ops
from ..utils.benchmarks import *
from ..utils.common import *

from .helpers import get_dtypes

__all__ = ['get_empty_blob', 'get_ones_blob', 'get_zeros_blob', 'get_ones_like_blob', 'get_zeros_like_blob',
           'get_arange_blob', 'get_linspace_blob']


# creation ops
def get_empty_blob(dtypes=AllTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Empty,
            config_func,
            run_creation_op_benchmark), 'empty'


def get_ones_blob(dtypes=AllTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Ones,
            config_func,
            run_creation_op_benchmark), 'ones'


def get_zeros_blob(dtypes=AllTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Zeros,
            config_func,
            run_creation_op_benchmark), 'zeros'


def get_ones_like_blob(dtypes=AllTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Ones_like,
            config_func,
            run_unary_op_benchmark), 'ones_like'


def get_zeros_like_blob(dtypes=AllTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Zeros_like,
            config_func,
            run_unary_op_benchmark), 'zeros_like'


def get_arange_blob(dtypes=AllTypes, is_random=True):
    return (ops.Arange,
            partial(get_range_creation_config, 'arange', get_dtypes(dtypes)),
            run_creation_op_benchmark), 'arange'


def get_linspace_blob(dtypes=AllTypes, is_random=True):
    return (ops.Linspace,
            partial(get_range_creation_config, 'linspace', get_dtypes(dtypes)),
            run_creation_op_benchmark), 'linspace'
