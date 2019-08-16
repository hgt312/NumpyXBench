from functools import partial

from ..configs import *
from .. import operators as ops
from ..utils.benchmarks import *
from ..utils.common import *

__all__ = ['add_blob', 'subtract_blob', 'multiply_blob', 'divide_blob', 'mod_blob', 'empty_blob', 'ones_blob',
           'zeros_blob', 'ones_like_blob', 'zeros_like_blob', 'arange_blob', 'linspace_blob', 'sum_blob',
           'prod_blob']

# elemwise binary ops
add_blob = (ops.Add, partial(get_random_size_config, AllTypes), run_binary_op_benchmark)
subtract_blob = (ops.Subtract, partial(get_random_size_config, AllTypes), run_binary_op_benchmark)
multiply_blob = (ops.Multiply, partial(get_random_size_config, AllTypes), run_binary_op_benchmark)
divide_blob = (ops.Divide, partial(get_random_size_config, AllTypes), run_binary_op_benchmark)
mod_blob = (ops.Mod, partial(get_random_size_config, AllTypes), run_binary_op_benchmark)

# creation ops
empty_blob = (ops.Empty, partial(get_random_size_config, AllTypes), run_creation_op_benchmark)
ones_blob = (ops.Ones, partial(get_random_size_config, AllTypes), run_creation_op_benchmark)
zeros_blob = (ops.Zeros, partial(get_random_size_config, AllTypes), run_creation_op_benchmark)
ones_like_blob = (ops.Ones_like, partial(get_random_size_config, AllTypes), run_unary_op_benchmark)
zeros_like_blob = (ops.Zeros_like, partial(get_random_size_config, AllTypes), run_unary_op_benchmark)
arange_blob = (ops.Arange, partial(get_range_creation_config, 'arange', AllTypes), run_creation_op_benchmark)
linspace_blob = (ops.Linspace, partial(get_range_creation_config, 'linspace', AllTypes), run_creation_op_benchmark)

# one input with axis ops
sum_blob = (ops.Sum, partial(get_random_withaxis_config, AllTypes), run_withaxis_unary_benchmark)
prod_blob = (ops.Prod, partial(get_random_withaxis_config, AllTypes), run_withaxis_unary_benchmark)
