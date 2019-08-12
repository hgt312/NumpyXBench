import inspect
import sys

try:
    import mxnet
    import torch
except ImportError:
    pass
import numpy
from jinja2 import Template

from ..utils.common import backend_switcher

__all__ = []


class RandomOp(object):
    """
    Random operator class, used to generate classes of operator which
    under namespace `numpy.random` automatically.
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
            module = sys.modules['.'.join([backend_switcher[self._backend], 'random'])]
        except AttributeError:
            raise Warning(f'Backend: {self._backend} not support or not installed!')

        return getattr(module, self._name)


template_code = """
class {{ name | capitalize }}(RandomOp):
    def __init__(self, backend):
        super({{ name | capitalize }}, self).__init__(backend=backend, name='{{ name }}')
"""
template = Template(template_code)


def _gen_random_op_list():
    random_op_list = []
    for obj_name in dir(numpy.random):
        obj = getattr(numpy.random, obj_name)
        if inspect.isbuiltin(obj):
            random_op_list.append(obj_name)
    random_op_list = [i for i in random_op_list if not i.startswith('_')]
    random_op_list = [i for i in random_op_list if not i[0].isupper()]
    return random_op_list


def _gen_random_op_class(name):
    local = {}
    exec(template.render(name=name), None, local)  # pylint: disable=exec-used
    op_name = name.capitalize()
    op_class = local[op_name]
    op_module = 'NumpyXBench.operators.random_ops'
    current_module = sys.modules[op_module]
    setattr(current_module, op_name, op_class)
    current_module.__all__.append(op_name)


for op in _gen_random_op_list():
    if op not in __all__:
        _gen_random_op_class(op)
