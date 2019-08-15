from .blobs import *
from . import operators
from .utils.mxnet_util import *


def test_mxnet_coverage():
    res = {'passed': [], 'failed': []}
    op_list = [i for i in dir(operators) if i[0].isupper() and i != 'CommonOp']
    print('Start MXNet NumPy coverage test!')
    print('#' * 60)
    for op_name in op_list:
        op = getattr(operators, op_name)('mx')
        flag = True
        try:
            op.get_forward_func()
            res['passed'].append(op_name.lower())
        except (AttributeError, KeyError, Warning) as e:
            flag = False
            res['failed'].append(op_name.lower())
        print("'{0}' under {1} check {2}.".format(op_name.lower(),
                                                  op.__module__.split('.')[-1],
                                                  'passed' if flag else 'failed'))
    print('#' * 60)
    print('End MXNet NumPy coverage test!')
    return res
