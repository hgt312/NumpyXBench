from ..configs import *
from .. import operators as ops
from ..utils import *

add_blob = (ops.Add, get_random_size_config, run_binary_op_benchmark)
subtract_blob = (ops.Subtract, get_random_size_config, run_binary_op_benchmark)
multiply_blob = (ops.Multiply, get_random_size_config, run_binary_op_benchmark)
divide_blob = (ops.Divide, get_random_size_config, run_binary_op_benchmark)
