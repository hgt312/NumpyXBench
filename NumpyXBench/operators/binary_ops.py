import sys

try:
    import numpy
    import mxnet
    import torch
except ImportError:
    pass

__all__ = ['Add', 'Subtract', 'Multiply', 'Divide']


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


class Add(CommonOp):
    def __init__(self, backend):
        super(Add, self).__init__(backend=backend, name='add')


class Subtract(CommonOp):
    def __init__(self, backend):
        super(Subtract, self).__init__(backend=backend, name='subtract')


class Multiply(CommonOp):
    def __init__(self, backend):
        super(Multiply, self).__init__(backend=backend, name='multiply')


class Divide(CommonOp):
    def __init__(self, backend):
        super(Divide, self).__init__(backend=backend, name='divide')
