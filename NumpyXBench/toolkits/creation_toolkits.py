from ..configs import *
from .. import operators as ops
from ..utils.benchmarks import *

from .toolkit import Toolkit

__all__ = ['empty_toolkit', 'ones_toolkit', 'zeros_toolkit', 'ones_like_toolkit', 'zeros_like_toolkit',
           'arange_toolkit', 'linspace_toolkit']


# creation ops
empty_toolkit = Toolkit(has_backward=False, name='empty', operator_cls=ops.Empty,
                        random_config_func=get_random_size_config,
                        determined_config_func=get_size_configs,
                        benchmark_func=run_creation_op_benchmark)

ones_toolkit = Toolkit(has_backward=False, name='ones', operator_cls=ops.Ones,
                       random_config_func=get_random_size_config,
                       determined_config_func=get_size_configs,
                       benchmark_func=run_creation_op_benchmark)

zeros_toolkit = Toolkit(has_backward=False, name='zeros', operator_cls=ops.Zeros,
                        random_config_func=get_random_size_config,
                        determined_config_func=get_size_configs,
                        benchmark_func=run_creation_op_benchmark)

ones_like_toolkit = Toolkit(has_backward=False, name='ones_like', operator_cls=ops.Ones_like,
                            random_config_func=get_random_size_config,
                            determined_config_func=get_size_configs,
                            benchmark_func=run_unary_op_benchmark)

zeros_like_toolkit = Toolkit(has_backward=False, name='zeros_like', operator_cls=ops.Zeros_like,
                             random_config_func=get_random_size_config,
                             determined_config_func=get_size_configs,
                             benchmark_func=run_unary_op_benchmark)

arange_toolkit = Toolkit(has_backward=False, name='arange', operator_cls=ops.Arange,
                         random_config_func=get_random_arange_config,
                         benchmark_func=run_creation_op_benchmark)

linspace_toolkit = Toolkit(has_backward=False, name='linspace', operator_cls=ops.Linspace,
                           random_config_func=get_random_linspace_config,
                           benchmark_func=run_creation_op_benchmark)
