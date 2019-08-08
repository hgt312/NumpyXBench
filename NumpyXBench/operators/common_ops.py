import sys

try:
    import numpy
    import mxnet
    import torch
except ImportError:
    pass
from jinja2 import Template

__all__ = []

common_op_list = ['add', 'subtract', 'multiply', 'divide']


class CommonOp(object):
    def __init__(self, backend='numpy', name=None):
        self._backend = backend
        self._name = name

    def get_backend(self):
        return self._backend

    def get_forward_func(self, *args, **kwargs):
        if self._backend in ['numpy', 'np']:
            module = sys.modules['numpy']
        elif self._backend in ['mxnet', 'mx']:
            module = sys.modules['mxnet.numpy']
        elif self._backend in ['pytorch', 'torch']:
            module = sys.modules['torch']
        else:
            raise NotImplementedError("Backend not supported now!")
        return getattr(module, self._name)


template_code = """
class {{ name | capitalize }}(CommonOp):
    def __init__(self, backend):
        super(Add, self).__init__(backend=backend, name='{{ name }}')
"""
template = Template(template_code)


def _gen_binary_op_class(name):
    local = {}
    exec(template.render(name=name), None, local)  # pylint: disable=exec-used
    op_name = name.capitalize()
    op_class = local[op_name]
    op_module = 'NumpyXBench.operators.common_ops'
    current_module = sys.modules[op_module]
    setattr(current_module, op_name, op_class)
    current_module.__all__.append(op_name)


for op in common_op_list:
    _gen_binary_op_class(op)
