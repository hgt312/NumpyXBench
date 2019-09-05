from ..configs import *
from .. import operators as ops
from ..utils.benchmarks import *

from .toolkit import Toolkit

__all__ = ['sum_toolkit', 'prod_toolkit']


# reduction ops
sum_toolkit = Toolkit(has_backward=True, name='sum', operator_cls=ops.Sum,
                      random_config_func=get_random_withaxis_config,
                      determined_config_func=get_size_axis_configs,
                      benchmark_func=run_unary_op_benchmark)

prod_toolkit = Toolkit(has_backward=True, name='prod', operator_cls=ops.Prod,
                       random_config_func=get_random_withaxis_config,
                       determined_config_func=get_size_axis_configs,
                       benchmark_func=run_unary_op_benchmark)
