from functools import partial

from ..configs import *
from .. import operators as ops
from ..utils.benchmarks import *
from ..utils.common import *

__all__ = ['get_add_blob', 'get_subtract_blob', 'get_multiply_blob', 'get_divide_blob', 'get_mod_blob',
           'get_empty_blob', 'get_ones_blob', 'get_zeros_blob', 'get_ones_like_blob', 'get_zeros_like_blob',
           'get_arange_blob', 'get_linspace_blob', 'get_sum_blob', 'get_prod_blob']


# helper functions
def get_dtypes(dtypes):
    if not (isinstance(dtypes, str) or isinstance(dtypes, list)):
        raise AttributeError("Dtypes must be a string or list!")
    elif isinstance(dtypes, str):
        if dtypes in ['real_types', 'RealTypes']:
            return RealTypes
        elif dtypes in ['all_types', 'AllTypes']:
            return AllTypes
        else:
            raise AttributeError('Unknown dtypes name!')
    else:
        return dtypes


# elemwise binary ops
def get_add_blob(dtypes=RealTypes):
    return (ops.Add,
            partial(get_random_size_config, get_dtypes(dtypes)),
            run_binary_op_benchmark), 'add'


def get_subtract_blob(dtypes=RealTypes):
    return (ops.Subtract,
            partial(get_random_size_config, get_dtypes(dtypes)),
            run_binary_op_benchmark), 'subtract'


def get_multiply_blob(dtypes=RealTypes):
    return (ops.Multiply,
            partial(get_random_size_config, get_dtypes(dtypes)),
            run_binary_op_benchmark), 'multiply'


def get_divide_blob(dtypes=RealTypes):
    return (ops.Divide,
            partial(get_random_size_config, get_dtypes(dtypes)),
            run_binary_op_benchmark), 'divide'


def get_mod_blob(dtypes=RealTypes):
    return (ops.Mod,
            partial(get_random_size_config, get_dtypes(dtypes)),
            run_binary_op_benchmark), 'mod'


# creation ops
def get_empty_blob(dtypes=AllTypes):
    return (ops.Empty,
            partial(get_random_size_config, get_dtypes(dtypes)),
            run_creation_op_benchmark), 'empty'


def get_ones_blob(dtypes=AllTypes):
    return (ops.Ones,
            partial(get_random_size_config, get_dtypes(dtypes)),
            run_creation_op_benchmark), 'ones'


def get_zeros_blob(dtypes=AllTypes):
    return (ops.Zeros,
            partial(get_random_size_config, get_dtypes(dtypes)),
            run_creation_op_benchmark), 'zeros'


def get_ones_like_blob(dtypes=AllTypes):
    return (ops.Ones_like,
            partial(get_random_size_config, get_dtypes(dtypes)),
            run_unary_op_benchmark), 'ones_like'


def get_zeros_like_blob(dtypes=AllTypes):
    return (ops.Zeros_like,
            partial(get_random_size_config, get_dtypes(dtypes)),
            run_unary_op_benchmark), 'zeros_like'


def get_arange_blob(dtypes=AllTypes):
    return (ops.Arange,
            partial(get_range_creation_config, 'arange', get_dtypes(dtypes)),
            run_creation_op_benchmark), 'arange'


def get_linspace_blob(dtypes=AllTypes):
    return (ops.Linspace,
            partial(get_range_creation_config, 'linspace', get_dtypes(dtypes)),
            run_creation_op_benchmark), 'linspace'


# one input with axis ops
def get_sum_blob(dtypes=RealTypes):
    return (ops.Sum,
            partial(get_random_withaxis_config, get_dtypes(dtypes)),
            run_withaxis_unary_benchmark), 'sum'


def get_prod_blob(dtypes=RealTypes):
    return (ops.Prod,
            partial(get_random_withaxis_config, get_dtypes(dtypes)),
            run_withaxis_unary_benchmark), 'prod'
