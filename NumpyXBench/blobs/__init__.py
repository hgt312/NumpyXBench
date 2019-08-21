from functools import partial

from ..configs import *
from .. import operators as ops
from ..utils.benchmarks import *
from ..utils.common import *

from .helpers import get_dtypes
from .elemwise_binary_blobs import *
from .creation_blobs import *

__all__ = ['get_add_blob', 'get_subtract_blob', 'get_multiply_blob', 'get_divide_blob', 'get_mod_blob',
           'get_empty_blob', 'get_ones_blob', 'get_zeros_blob', 'get_ones_like_blob', 'get_zeros_like_blob',
           'get_arange_blob', 'get_linspace_blob', 'get_sum_blob', 'get_prod_blob']


# one input with axis ops
def get_sum_blob(dtypes=RealTypes, is_random=True):
    return (ops.Sum,
            partial(get_random_withaxis_config, get_dtypes(dtypes)),
            run_withaxis_unary_benchmark), 'sum'


def get_prod_blob(dtypes=RealTypes, is_random=True):
    return (ops.Prod,
            partial(get_random_withaxis_config, get_dtypes(dtypes)),
            run_withaxis_unary_benchmark), 'prod'
