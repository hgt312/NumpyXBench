from . import blobs
from . import operators
from .utils.common import backend_switcher
from .utils.benchmarks import run_op_frameworks_benchmark

__all__ = ['test_numpy_coverage', 'test_all_blobs']


def test_numpy_coverage(backend_name):
    backend = backend_switcher[backend_name]
    res = {'passed': [], 'failed': []}
    op_list = [i for i in dir(operators) if i[0].isupper()]
    print('Start {0} coverage test!'.format(backend))
    print('#' * 60)
    for op_name in op_list:
        op = getattr(operators, op_name)(backend_name)
        flag = True
        try:
            op.get_forward_func()
            res['passed'].append((op_name.lower(), op.__module__.split('.')[-1]))
        except (AttributeError, KeyError, Warning) as e:
            flag = False
            res['failed'].append((op_name.lower(), op.__module__.split('.')[-1]))
        print("'{0}' under {1} check {2}.".format(op_name.lower(),
                                                  op.__module__.split('.')[-1],
                                                  'passed' if flag else 'failed'))
    print('#' * 60)
    print('End {0} coverage test!'.format(backend))
    return res


def test_all_blobs(dtypes='RealTypes', mode='forward'):
    backends = ['chainerx', 'jax.numpy', 'mxnet.numpy', 'numpy']
    blobs_list = blobs.__all__
    blobs_list = [getattr(blobs, i) for i in blobs_list]
    result = {}
    for blob_func in blobs_list:
        blob, name = blob_func(dtypes)
        result[name] = run_op_frameworks_benchmark(*blob, backends, mode)
    return result
