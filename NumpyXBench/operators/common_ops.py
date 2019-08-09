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

module_switcher = {
    'numpy': 'numpy',
    'np': 'numpy',
    'mxnet': 'mxnet.numpy',
    'pytorch': 'torch',
    'torch': 'torch',
}


class CommonOp(object):
    """
    Common operator class, used to generate classes of operator which
    under namespace `numpy` automatically.
    """

    def __init__(self, backend='numpy', name=None):
        """
        Init function

        Parameters
        ----------
        backend : str
            Name of the backend.
        name : str
            Name of the operator.
        """
        self._backend = backend
        self._name = name

    def get_backend(self):
        """
        Get the current used backend of the Op.
        """
        return self._backend

    def get_forward_func(self):
        """
        Get the forward function of the Op.
        """
        try:
            module = sys.modules[module_switcher[self._backend]]
        except ValueError:
            raise NotImplementedError(f'Backend: {self._backend} not supported now!')

        return getattr(module, self._name)


template_code = """
class {{ name | capitalize }}(CommonOp):
    def __init__(self, backend):
        super({{ name | capitalize }}, self).__init__(backend=backend, name='{{ name }}')
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
