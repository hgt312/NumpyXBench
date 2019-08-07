from ..configs import *
from .. import operators as ops
from ..utils import *

add_blob = (ops.Add, get_random_size_config)
subtract_blob = (ops.Subtract, get_random_size_config)
multiply_blob = (ops.Multiply, get_random_size_config)
divide_blob = (ops.Divide, get_random_size_config)
