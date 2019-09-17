from ..configs import *
from .. import operators as ops
from ..utils.benchmarks import *

from .toolkit import Toolkit

__all__ = ['add_toolkit', 'subtract_toolkit', 'multiply_toolkit', 'divide_toolkit', 'mod_toolkit']


# elemwise binary ops
add_toolkit = Toolkit(has_backward=True, operator_cls=ops.Add,
                      random_config_func=get_random_size_config,
                      determined_config_func=get_size_configs,
                      benchmark_func=run_binary_op_benchmark)

subtract_toolkit = Toolkit(has_backward=True, operator_cls=ops.Subtract,
                           random_config_func=get_random_size_config,
                           determined_config_func=get_size_configs,
                           benchmark_func=run_binary_op_benchmark)

multiply_toolkit = Toolkit(has_backward=True, operator_cls=ops.Multiply,
                           random_config_func=get_random_size_config,
                           determined_config_func=get_size_configs,
                           benchmark_func=run_binary_op_benchmark)

divide_toolkit = Toolkit(has_backward=True, operator_cls=ops.Divide,
                         random_config_func=get_random_size_config,
                         determined_config_func=get_size_configs,
                         benchmark_func=run_binary_op_benchmark)

mod_toolkit = Toolkit(has_backward=True, operator_cls=ops.Mod,
                      random_config_func=get_random_size_config,
                      determined_config_func=get_size_configs,
                      benchmark_func=run_binary_op_benchmark)

broadcast_add_toolkit = Toolkit(has_backward=True, operator_cls=ops.Add,
                                random_config_func=get_random_size_config,
                                determined_config_func=get_broadcast_configs,
                                benchmark_func=run_binary_broadcast_op_benchmark)
broadcast_add_toolkit.name = "broadcast_add"

broadcast_subtract_toolkit = Toolkit(has_backward=True, operator_cls=ops.Subtract,
                                     random_config_func=get_random_size_config,
                                     determined_config_func=get_broadcast_configs,
                                     benchmark_func=run_binary_broadcast_op_benchmark)
broadcast_subtract_toolkit.name = "broadcast_subtract"

broadcast_multiply_toolkit = Toolkit(has_backward=True, operator_cls=ops.Multiply,
                                     random_config_func=get_random_size_config,
                                     determined_config_func=get_broadcast_configs,
                                     benchmark_func=run_binary_broadcast_op_benchmark)
broadcast_multiply_toolkit.name = "broadcast_multiply"

broadcast_divide_toolkit = Toolkit(has_backward=True, operator_cls=ops.Divide,
                                   random_config_func=get_random_size_config,
                                   determined_config_func=get_broadcast_configs,
                                   benchmark_func=run_binary_broadcast_op_benchmark)
broadcast_divide_toolkit.name = "broadcast_divide"

broadcast_mod_toolkit = Toolkit(has_backward=True, operator_cls=ops.Mod,
                                random_config_func=get_random_size_config,
                                determined_config_func=get_broadcast_configs,
                                benchmark_func=run_binary_broadcast_op_benchmark)
broadcast_mod_toolkit.name = "broadcast_mod"
