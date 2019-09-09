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

from ..utils.common import backend_switcher

__all__ = []


class CommonOp(object):
    """
    Common operator class, used to generate classes of operator which
    under namespace `numpy` automatically.
    """
    _name = None

    def __init__(self, backend='numpy'):
        """
        Init function

        Parameters
        ----------
        backend : str
            Name of the backend.
        """
        self._backend = backend

    def get_backend(self):
        """
        Get the current used backend of the Op.
        """
        return self._backend

    @classmethod
    def get_name(cls):
        """
        Get the name of the Op.
        """
        return cls._name

    def get_forward_func(self):
        """
        Get the forward function of the Op.
        """
        try:
            module = sys.modules[backend_switcher[self._backend]]
        except (AttributeError, KeyError) as e:
            return None
        if hasattr(module, self._name):
            return getattr(module, self._name)
        else:
            return None


template_code = """
class {{ name | capitalize }}(CommonOp):
    _name = '{{ name }}'
    def __init__(self, backend):
        super({{ name | capitalize }}, self).__init__(backend=backend)
"""
template = Template(template_code)


def _gen_common_op_list():
    common_op_list = []
    for obj_name in dir(numpy):
        obj = getattr(numpy, obj_name)
        if any([inspect.isbuiltin(obj),
                inspect.isfunction(obj),
                isinstance(obj, numpy.ufunc)]):
            common_op_list.append(obj_name)
    common_op_list = [i for i in common_op_list if not i.startswith('_')]
    return common_op_list


def _gen_common_op_class(name):
    local = {}
    exec(template.render(name=name), None, local)  # pylint: disable=exec-used
    op_name = name.capitalize()
    op_class = local[op_name]
    op_module = 'NumpyXBench.operators.common_ops'
    current_module = sys.modules[op_module]
    setattr(current_module, op_name, op_class)
    current_module.__all__.append(op_name)


for op in _gen_common_op_list():
    if op not in __all__:
        _gen_common_op_class(op)
