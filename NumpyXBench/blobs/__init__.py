from functools import partial

from ..configs import *
from .. import operators as ops
from ..utils.benchmarks import *
from ..utils.common import *

from .helpers import get_dtypes
from .elemwise_binary_blobs import *
from .creation_blobs import *
from .unary_blobs import *

__all__ = ['get_add_blob', 'get_subtract_blob', 'get_multiply_blob', 'get_divide_blob', 'get_mod_blob',
           'get_empty_blob', 'get_ones_blob', 'get_zeros_blob', 'get_ones_like_blob', 'get_zeros_like_blob',
           'get_arange_blob', 'get_linspace_blob', 'get_sum_blob', 'get_prod_blob', 'get_abs_blob', 'get_arccos_blob',
           'get_arccosh_blob', 'get_arcsin_blob', 'get_arcsinh_blob', 'get_arctan_blob', 'get_arctanh_blob',
           'get_cbrt_blob', 'get_ceil_blob', 'get_cos_blob', 'get_cosh_blob', 'get_degrees_blob', 'get_exp_blob',
           'get_expm1_blob', 'get_fix_blob', 'get_floor_blob', 'get_log1p_blob', 'get_log2_blob', 'get_log10_blob',
           'get_log_blob', 'get_logical_not_blob', 'get_radians_blob', 'get_reciprocal_blob', 'get_rint_blob',
           'get_sign_blob', 'get_sin_blob', 'get_sinh_blob', 'get_sqrt_blob', 'get_square_blob', 'get_tan_blob',
           'get_tanh_blob', 'get_trunc_blob']


# one input with axis ops
def get_sum_blob(dtypes=RealTypes, is_random=True):
    return (ops.Sum,
            partial(get_random_withaxis_config, get_dtypes(dtypes)),
            run_withaxis_unary_benchmark), 'sum'


def get_prod_blob(dtypes=RealTypes, is_random=True):
    return (ops.Prod,
            partial(get_random_withaxis_config, get_dtypes(dtypes)),
            run_withaxis_unary_benchmark), 'prod'
