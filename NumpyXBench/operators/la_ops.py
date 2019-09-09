import inspect
import sys

try:
    import mxnet
    import jax
    import chainerx
except Exception:
    pass
import numpy
from jinja2 import Template

from .common_ops import CommonOp
from ..utils.common import backend_switcher

__all__ = []


class LAOp(CommonOp):
    """
    Linear algebra operator class, used to generate classes of operator which
    under namespace `numpy.linalg` automatically.
    """

    def get_forward_func(self):
        """
        Get the forward function of the Op.
        """
        try:
            module = sys.modules['.'.join([backend_switcher[self._backend], 'linalg'])]
        except (AttributeError, KeyError) as e:
            return None
        if hasattr(module, self._name):
            return getattr(module, self._name)
        else:
            return None


template_code = """
class {{ name | capitalize }}(LAOp):
    _name = '{{ name }}'
    def __init__(self, backend):
        super({{ name | capitalize }}, self).__init__(backend=backend)
"""
template = Template(template_code)


def _gen_la_op_list():
    la_op_list = []
    for obj_name in dir(numpy.linalg):
        obj = getattr(numpy.linalg, obj_name)
        if inspect.isfunction(obj):
            la_op_list.append(obj_name)
    la_op_list = [i for i in la_op_list if not i.startswith('_')]
    return la_op_list


def _gen_la_op_class(name):
    local = {}
    exec(template.render(name=name), None, local)  # pylint: disable=exec-used
    op_name = name.capitalize()
    op_class = local[op_name]
    op_module = 'NumpyXBench.operators.la_ops'
    current_module = sys.modules[op_module]
    setattr(current_module, op_name, op_class)
    current_module.__all__.append(op_name)


for op in _gen_la_op_list():
    if op not in __all__:
        _gen_la_op_class(op)
