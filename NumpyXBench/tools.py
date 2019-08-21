from itertools import chain

try:
    import mxnet
    import torch
    import chainerx
    import jax
except Exception:
    pass

from bokeh.io import show, output_file, output_notebook
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

from . import blobs
from . import operators
from .utils.common import backend_switcher
from .utils.benchmarks import run_op_frameworks_benchmark

__all__ = ['test_numpy_coverage', 'test_all_blobs', 'draw_one_plot']


def global_set_gpu():
    mxnet.test_utils.set_default_context(mxnet.gpu(0))
    chainerx.set_default_device('cuda:0')


def global_set_cpu():
    mxnet.test_utils.set_default_context(mxnet.cpu())
    chainerx.set_default_device('native')


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


def test_all_blobs(dtypes='RealTypes', mode='forward', is_random=True, times=6, warmup=10, runs=25):
    backends = ['chainerx', 'jax.numpy', 'mxnet.numpy', 'numpy']
    blobs_list = blobs.__all__
    blobs_list = [getattr(blobs, i) for i in blobs_list]
    result = {}
    for blob_func in blobs_list:
        blob, name = blob_func(dtypes, is_random)
        result[name] = run_op_frameworks_benchmark(*blob, backends, mode, is_random, times, warmup, runs)
    return result


def test_blobs(blobs_list, dtypes='RealTypes', mode='forward', is_random=True, times=6, warmup=10, runs=25):
    backends = ['chainerx', 'jax.numpy', 'mxnet.numpy', 'numpy']
    result = {}
    for blob_func in blobs_list:
        blob, name = blob_func(dtypes, is_random)
        result[name] = run_op_frameworks_benchmark(*blob, backends, mode, is_random, times, warmup, runs)
    return result


def draw_one_plot(name, data, mode="file", filename="demo.html"):
    num = len(data)
    x_labels = ['config{0}'.format(i + 1) for i in range(num)]
    backends = ['numpy', 'mxnet', 'jax', 'chainerx']
    x = [(l, b) for l in x_labels for b in backends]

    if mode == "file":
        output_file(filename)
    else:
        output_notebook()
    palette = ["#756bb1", "#43a2ca", "#e84d60", "#2ca25f"]
    tooltips = [("config", "@configs"), ("seconds", "@seconds"), ("rate", "@rates")]

    configs = list(chain.from_iterable([str(d['config'])] * 4 for d in data))
    seconds = list(chain.from_iterable((d['numpy'], d['mxnet.numpy'], d['jax.numpy'], d['chainerx']) for d in data))
    rates = list(chain.from_iterable((1.,
                                      d['numpy'] / d['mxnet.numpy'],
                                      d['numpy'] / d['jax.numpy'],
                                      d['numpy'] / d['chainerx']) for d in data))
    source = ColumnDataSource(data=dict(x=x, configs=configs, seconds=seconds, rates=rates))
    p = figure(x_range=FactorRange(*x),
               plot_height=600, plot_width=800,
               title="NumPy operator {0}".format(name), y_axis_label="Speed rate",
               tooltips=tooltips)
    p.vbar(x='x', top='rates', source=source, width=0.9, bottom=-0.25, line_color="white",
           fill_color=factor_cmap('x', palette=palette, factors=backends, start=1, end=2))
    p.y_range.start = -0.25
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    show(p)
