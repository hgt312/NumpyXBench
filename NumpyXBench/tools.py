import argparse
import gc
from itertools import chain
import pprint
import os

try:
    import mxnet
    import chainerx
    import jax
except Exception:
    pass

from bokeh.io import save, show, output_file, output_notebook
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

from . import toolkits
from . import operators
from .utils.common import backend_switcher
from .utils.benchmarks import run_op_frameworks_benchmark

__all__ = ['test_numpy_coverage', 'test_all_operators', 'draw_one_plot', 'test_operators', 'generate_operator_reports',
           'global_set_cpu', 'global_set_gpu', 'generate_one_report']


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


def test_all_operators(dtypes='RealTypes', mode='forward', is_random=True, times=6, warmup=10, runs=25):
    backends = ['chainerx', 'jax.numpy', 'mxnet.numpy', 'numpy']
    toolkit_list = dir(toolkits)
    toolkit_list = [i for i in toolkit_list if i.endswith('_toolkit')]
    toolkit_list = [getattr(toolkits, i) for i in toolkit_list]
    result = {}
    for toolkit in toolkit_list:
        name = toolkit.get_name()
        result[name] = run_op_frameworks_benchmark(*toolkit.get_tools(dtypes, is_random),
                                                   backends, mode, times, warmup, runs)
        print("Done benchmark for `{0}`!".format(name))
    return result


def test_operators(toolkit_list, dtypes='RealTypes', mode='forward', is_random=True, times=6, warmup=10, runs=25):
    backends = ['chainerx', 'jax.numpy', 'mxnet.numpy', 'numpy']
    result = {}
    for toolkit in toolkit_list:
        name = toolkit.get_name()
        result[name] = run_op_frameworks_benchmark(*toolkit.get_tools(dtypes, is_random),
                                                   backends, mode, times, warmup, runs)
        print("Done benchmark for `{0}`!".format(name))
    return result


def draw_one_plot(name, data, mode="file", filename="demo.html", info=None):
    title = "NumPy operator {0}".format(name)
    if info:
        title += " - {0}".format(info)
    num = len(data)
    x_labels = ['config{0}'.format(i + 1) for i in range(num)]
    backends = ['numpy', 'mxnet', 'jax', 'chainerx']
    x = [(l, b) for l in x_labels for b in backends]

    if mode == "file":
        output_file(filename)
    else:
        output_notebook()
    palette = ["#756bb1", "#43a2ca", "#e84d60", "#2ca25f"]
    tooltips = [("config", "@configs"), ("millisecond", "@millisecond"), ("speedup", "@rates")]

    configs = list(chain.from_iterable([pprint.pformat(d['config'], width=1)] * 4 for d in data))
    millisecond = list(chain.from_iterable((d['numpy'], d['mxnet.numpy'], d['jax.numpy'], d['chainerx']) for d in data))
    rates = list(chain.from_iterable((1.,
                                      d['numpy'] / d['mxnet.numpy'] if d['mxnet.numpy'] else -1,
                                      d['numpy'] / d['jax.numpy'] if d['jax.numpy'] else -1,
                                      d['numpy'] / d['chainerx'] if d['chainerx'] else -1) for d in data))
    offset = -max(rates) / 15
    rates = [r if r > 0 else offset for r in rates]
    source = ColumnDataSource(data=dict(x=x, configs=configs, millisecond=millisecond, rates=rates))
    p = figure(x_range=FactorRange(*x),
               plot_height=600, plot_width=800,
               title=title, y_axis_label="Speedup",
               tooltips=tooltips,
               toolbar_location="above")
    p.vbar(x='x', top='rates', source=source, width=0.9, bottom=offset, line_color="white",
           fill_color=factor_cmap('x', palette=palette, factors=backends, start=1, end=2))
    p.y_range.start = offset
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    if mode == "file":
        save(p)
    else:
        show(p)


def use_html_template(filename):
    with open(filename, mode="r") as f:
        html = f.readlines()
    html[-1] += '\n'
    html = ["    " + h for h in html]
    html.insert(0, ".. raw:: html")
    with open(filename, mode="w") as f:
        f.writelines(html)


def generate_one_report(toolkit_name, warmup, runs, info):
    toolkit = getattr(toolkits, toolkit_name)
    base_path = os.path.dirname(os.path.abspath(__file__))
    backends = ['chainerx', 'jax.numpy', 'mxnet.numpy', 'numpy']
    op_name = toolkit.get_name()
    content = """Operator `{0}`
==========={1}

""".format(op_name, '=' * len(op_name))
    for dtype in toolkit.get_forward_dtypes():
        html_filename = "{0}_f_{1}.html".format(op_name, dtype)
        html_file = os.path.join(base_path, '../doc/_static/temp', html_filename)
        content += ".. include:: /_static/temp/{0}\n\n".format(html_filename)
        data = run_op_frameworks_benchmark(*toolkit.get_tools([dtype], False),
                                           backends, 'forward', 6, warmup, runs)
        draw_one_plot(op_name, data, mode='file', filename=html_file,
                      info=info + ", {0}, forward only".format(dtype) if info else None)
        use_html_template(html_file)
        gc.collect()
    if toolkit.get_backward_dtypes():
        for dtype in toolkit.get_backward_dtypes():
            html_filename = "{0}_b_{1}.html".format(op_name, dtype)
            html_file = os.path.join(base_path, '../doc/_static/temp', html_filename)
            content += ".. include:: /_static/temp/{0}\n\n".format(html_filename)
            data = run_op_frameworks_benchmark(*toolkit.get_tools([dtype], False),
                                               backends, 'both', 6, warmup, runs)
            draw_one_plot(op_name, data, mode='file', filename=html_file,
                          info=info + ", {0}, with backward".format(dtype) if info else None)
            use_html_template(html_file)
            gc.collect()
    rst_file = os.path.join(base_path, '../doc/reports', op_name + '.rst')
    with open(rst_file, mode='w') as f:
        f.write(content)
    print("Done report generation for `{0}`!".format(op_name))


def generate_operator_reports(warmup=10, runs=25, info=None):
    toolkit_list = dir(toolkits)
    toolkit_list = [i for i in toolkit_list if i.endswith('_toolkit')]
    for toolkit_name in toolkit_list:
        cmd_line = 'python3 -c "from NumpyXBench.tools import generate_one_report; ' \
              'generate_one_report(\'{0}\', {1}, {2}, \'{3}\')"'.format(toolkit_name, warmup, runs, info)
        os.system(cmd_line)
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Need parameters 'warmup' and 'runs'.")
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--runs", default=25, type=int)
    parser.add_argument("--info", default=None, type=str)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()
    if args.device == "gpu":
        global_set_gpu()
    else:
        global_set_cpu()
    generate_operator_reports(args.warmup, args.runs, args.info)
