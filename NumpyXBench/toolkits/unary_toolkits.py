from ..configs import *
from .. import operators as ops
from ..utils.benchmarks import *

from .toolkit import Toolkit

__all__ = ['abs_toolkit', 'arccos_toolkit', 'arccosh_toolkit', 'arcsin_toolkit', 'arcsinh_toolkit',
           'arctan_toolkit', 'arctanh_toolkit', 'cbrt_toolkit', 'ceil_toolkit', 'cos_toolkit', 'cosh_toolkit',
           'degrees_toolkit', 'exp_toolkit', 'expm1_toolkit', 'fix_toolkit', 'floor_toolkit', 'log_toolkit',
           'log10_toolkit', 'log1p_toolkit', 'log2_toolkit', 'logical_not_toolkit', 'radians_toolkit',
           'reciprocal_toolkit', 'rint_toolkit', 'sign_toolkit', 'sin_toolkit', 'sinh_toolkit', 'sqrt_toolkit',
           'square_toolkit', 'tan_toolkit', 'tanh_toolkit', 'trunc_toolkit']


# unary ops
abs_toolkit = Toolkit(has_backward=True, name='abs', operator_cls=ops.Abs,
                      random_config_func=get_random_size_config,
                      determined_config_func=get_size_configs,
                      benchmark_func=run_unary_op_benchmark)

arccos_toolkit = Toolkit(has_backward=True, name='arccos', operator_cls=ops.Arccos,
                         random_config_func=get_random_size_config,
                         determined_config_func=get_size_configs,
                         benchmark_func=run_unary_op_benchmark)

arccosh_toolkit = Toolkit(has_backward=True, name='arccosh', operator_cls=ops.Arccosh,
                          random_config_func=get_random_size_config,
                          determined_config_func=get_size_configs,
                          benchmark_func=run_unary_op_benchmark)

arcsin_toolkit = Toolkit(has_backward=True, name='arcsin', operator_cls=ops.Arcsin,
                         random_config_func=get_random_size_config,
                         determined_config_func=get_size_configs,
                         benchmark_func=run_unary_op_benchmark)

arcsinh_toolkit = Toolkit(has_backward=True, name='arcsinh', operator_cls=ops.Arcsinh,
                          random_config_func=get_random_size_config,
                          determined_config_func=get_size_configs,
                          benchmark_func=run_unary_op_benchmark)

arctan_toolkit = Toolkit(has_backward=True, name='arctan', operator_cls=ops.Arctan,
                         random_config_func=get_random_size_config,
                         determined_config_func=get_size_configs,
                         benchmark_func=run_unary_op_benchmark)

arctanh_toolkit = Toolkit(has_backward=True, name='arctanh', operator_cls=ops.Arctanh,
                          random_config_func=get_random_size_config,
                          determined_config_func=get_size_configs,
                          benchmark_func=run_unary_op_benchmark)

cbrt_toolkit = Toolkit(has_backward=True, name='cbrt', operator_cls=ops.Cbrt,
                       random_config_func=get_random_size_config,
                       determined_config_func=get_size_configs,
                       benchmark_func=run_unary_op_benchmark)

ceil_toolkit = Toolkit(has_backward=True, name='ceil', operator_cls=ops.Ceil,
                       random_config_func=get_random_size_config,
                       determined_config_func=get_size_configs,
                       benchmark_func=run_unary_op_benchmark)

cos_toolkit = Toolkit(has_backward=True, name='cos', operator_cls=ops.Cos,
                      random_config_func=get_random_size_config,
                      determined_config_func=get_size_configs,
                      benchmark_func=run_unary_op_benchmark)

cosh_toolkit = Toolkit(has_backward=True, name='cosh', operator_cls=ops.Cosh,
                       random_config_func=get_random_size_config,
                       determined_config_func=get_size_configs,
                       benchmark_func=run_unary_op_benchmark)

degrees_toolkit = Toolkit(has_backward=True, name='degrees', operator_cls=ops.Degrees,
                          random_config_func=get_random_size_config,
                          determined_config_func=get_size_configs,
                          benchmark_func=run_unary_op_benchmark)

exp_toolkit = Toolkit(has_backward=True, name='exp', operator_cls=ops.Exp,
                      random_config_func=get_random_size_config,
                      determined_config_func=get_size_configs,
                      benchmark_func=run_unary_op_benchmark)

expm1_toolkit = Toolkit(has_backward=True, name='expm1', operator_cls=ops.Expm1,
                        random_config_func=get_random_size_config,
                        determined_config_func=get_size_configs,
                        benchmark_func=run_unary_op_benchmark)

fix_toolkit = Toolkit(has_backward=True, name='fix', operator_cls=ops.Fix,
                      random_config_func=get_random_size_config,
                      determined_config_func=get_size_configs,
                      benchmark_func=run_unary_op_benchmark)

floor_toolkit = Toolkit(has_backward=True, name='floor', operator_cls=ops.Floor,
                        random_config_func=get_random_size_config,
                        determined_config_func=get_size_configs,
                        benchmark_func=run_unary_op_benchmark)

log_toolkit = Toolkit(has_backward=True, name='log', operator_cls=ops.Log,
                      random_config_func=get_random_size_config,
                      determined_config_func=get_size_configs,
                      benchmark_func=run_unary_op_benchmark)

log10_toolkit = Toolkit(has_backward=True, name='log10', operator_cls=ops.Log10,
                        random_config_func=get_random_size_config,
                        determined_config_func=get_size_configs,
                        benchmark_func=run_unary_op_benchmark)

log1p_toolkit = Toolkit(has_backward=True, name='log1p', operator_cls=ops.Log1p,
                        random_config_func=get_random_size_config,
                        determined_config_func=get_size_configs,
                        benchmark_func=run_unary_op_benchmark)

log2_toolkit = Toolkit(has_backward=True, name='log2', operator_cls=ops.Log2,
                       random_config_func=get_random_size_config,
                       determined_config_func=get_size_configs,
                       benchmark_func=run_unary_op_benchmark)

logical_not_toolkit = Toolkit(has_backward=True, name='logical_not', operator_cls=ops.Logical_not,
                              random_config_func=get_random_size_config,
                              determined_config_func=get_size_configs,
                              benchmark_func=run_unary_op_benchmark)

radians_toolkit = Toolkit(has_backward=True, name='radians', operator_cls=ops.Radians,
                          random_config_func=get_random_size_config,
                          determined_config_func=get_size_configs,
                          benchmark_func=run_unary_op_benchmark)

reciprocal_toolkit = Toolkit(has_backward=True, name='reciprocal', operator_cls=ops.Reciprocal,
                             random_config_func=get_random_size_config,
                             determined_config_func=get_size_configs,
                             benchmark_func=run_unary_op_benchmark)

rint_toolkit = Toolkit(has_backward=True, name='rint', operator_cls=ops.Rint,
                       random_config_func=get_random_size_config,
                       determined_config_func=get_size_configs,
                       benchmark_func=run_unary_op_benchmark)

sign_toolkit = Toolkit(has_backward=True, name='sign', operator_cls=ops.Sign,
                       random_config_func=get_random_size_config,
                       determined_config_func=get_size_configs,
                       benchmark_func=run_unary_op_benchmark)

sin_toolkit = Toolkit(has_backward=True, name='sin', operator_cls=ops.Sin,
                      random_config_func=get_random_size_config,
                      determined_config_func=get_size_configs,
                      benchmark_func=run_unary_op_benchmark)

sinh_toolkit = Toolkit(has_backward=True, name='sinh', operator_cls=ops.Sinh,
                       random_config_func=get_random_size_config,
                       determined_config_func=get_size_configs,
                       benchmark_func=run_unary_op_benchmark)

sqrt_toolkit = Toolkit(has_backward=True, name='sqrt', operator_cls=ops.Sqrt,
                       random_config_func=get_random_size_config,
                       determined_config_func=get_size_configs,
                       benchmark_func=run_unary_op_benchmark)

square_toolkit = Toolkit(has_backward=True, name='square', operator_cls=ops.Square,
                         random_config_func=get_random_size_config,
                         determined_config_func=get_size_configs,
                         benchmark_func=run_unary_op_benchmark)

tan_toolkit = Toolkit(has_backward=True, name='tan', operator_cls=ops.Tan,
                      random_config_func=get_random_size_config,
                      determined_config_func=get_size_configs,
                      benchmark_func=run_unary_op_benchmark)

tanh_toolkit = Toolkit(has_backward=True, name='tanh', operator_cls=ops.Tanh,
                       random_config_func=get_random_size_config,
                       determined_config_func=get_size_configs,
                       benchmark_func=run_unary_op_benchmark)

trunc_toolkit = Toolkit(has_backward=True, name='trunc', operator_cls=ops.Trunc,
                        random_config_func=get_random_size_config,
                        determined_config_func=get_size_configs,
                        benchmark_func=run_unary_op_benchmark)
