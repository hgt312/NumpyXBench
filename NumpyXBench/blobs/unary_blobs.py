from functools import partial

from ..configs import *
from .. import operators as ops
from ..utils.benchmarks import *
from ..utils.common import *

from .helpers import get_dtypes

__all__ = ['get_abs_blob', 'get_arccos_blob', 'get_arccosh_blob', 'get_arcsin_blob', 'get_arcsinh_blob',
           'get_arctan_blob', 'get_arctanh_blob', 'get_cbrt_blob', 'get_ceil_blob', 'get_cos_blob',
           'get_cosh_blob', 'get_degrees_blob', 'get_exp_blob', 'get_expm1_blob', 'get_fix_blob', 'get_floor_blob',
           'get_log1p_blob', 'get_log2_blob', 'get_log10_blob', 'get_log_blob', 'get_logical_not_blob',
           'get_radians_blob', 'get_reciprocal_blob', 'get_rint_blob', 'get_sign_blob', 'get_sin_blob',
           'get_sinh_blob', 'get_sqrt_blob', 'get_square_blob', 'get_tan_blob', 'get_tanh_blob', 'get_trunc_blob']


# unary ops
def get_abs_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Abs,
            config_func,
            run_unary_op_benchmark), 'abs'


def get_arccos_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Arccos,
            config_func,
            run_unary_op_benchmark), 'arccos'


def get_arccosh_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Arccosh,
            config_func,
            run_unary_op_benchmark), 'arccosh'


def get_arcsin_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Arcsin,
            config_func,
            run_unary_op_benchmark), 'arcsin'


def get_arcsinh_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Arcsinh,
            config_func,
            run_unary_op_benchmark), 'arcsinh'


def get_arctan_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Arctan,
            config_func,
            run_unary_op_benchmark), 'arctan'


def get_arctanh_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Arctanh,
            config_func,
            run_unary_op_benchmark), 'arctanh'


def get_cbrt_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Cbrt,
            config_func,
            run_unary_op_benchmark), 'cbrt'


def get_ceil_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Ceil,
            config_func,
            run_unary_op_benchmark), 'ceil'


def get_cos_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Cos,
            config_func,
            run_unary_op_benchmark), 'cos'


def get_cosh_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Cosh,
            config_func,
            run_unary_op_benchmark), 'cosh'


def get_degrees_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Degrees,
            config_func,
            run_unary_op_benchmark), 'degrees'


def get_exp_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Exp,
            config_func,
            run_unary_op_benchmark), 'exp'


def get_expm1_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Expm1,
            config_func,
            run_unary_op_benchmark), 'expm1'


def get_fix_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Fix,
            config_func,
            run_unary_op_benchmark), 'fix'


def get_floor_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Floor,
            config_func,
            run_unary_op_benchmark), 'floor'


def get_log_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Log,
            config_func,
            run_unary_op_benchmark), 'log'


def get_log10_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Log10,
            config_func,
            run_unary_op_benchmark), 'log10'


def get_log1p_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Log1p,
            config_func,
            run_unary_op_benchmark), 'log1p'


def get_log2_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Log2,
            config_func,
            run_unary_op_benchmark), 'log2'


def get_logical_not_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Logical_not,
            config_func,
            run_unary_op_benchmark), 'logical_not'


def get_radians_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Radians,
            config_func,
            run_unary_op_benchmark), 'radians'


def get_reciprocal_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Reciprocal,
            config_func,
            run_unary_op_benchmark), 'reciprocal'


def get_rint_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Rint,
            config_func,
            run_unary_op_benchmark), 'rint'


def get_sign_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Sign,
            config_func,
            run_unary_op_benchmark), 'sign'


def get_sin_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Sin,
            config_func,
            run_unary_op_benchmark), 'sin'


def get_sinh_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Sinh,
            config_func,
            run_unary_op_benchmark), 'sinh'


def get_sqrt_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Sqrt,
            config_func,
            run_unary_op_benchmark), 'sqrt'


def get_square_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Square,
            config_func,
            run_unary_op_benchmark), 'square'


def get_tan_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Tan,
            config_func,
            run_unary_op_benchmark), 'tan'


def get_tanh_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Tanh,
            config_func,
            run_unary_op_benchmark), 'tanh'


def get_trunc_blob(dtypes=RealTypes, is_random=True):
    if is_random:
        config_func = partial(get_random_size_config, get_dtypes(dtypes))
    else:
        config_func = partial(get_size_configs, get_dtypes(dtypes))
    return (ops.Trunc,
            config_func,
            run_unary_op_benchmark), 'trunc'
