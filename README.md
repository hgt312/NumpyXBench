## What does this project do?

This is a project used to benchmark the operators (functions) of the python librarieswhich have compatible API with [Numpy](https://docs.scipy.org/doc/numpy/index.html), now it can generate some reports for operators in [MXNet](https://mxnet.apache.org/) (new numpy programming style), [ChainerX](https://docs.chainer.org/en/stable/chainerx/) and [JAX](https://github.com/google/jax).

## Operator coverage

I divide opertors into several categories: 

- Common operators, those can be found under `numpy`
- FFT operators, those can be found under `numpy.fft`
- Linear algebra operators, those can be found under `numpy.linalg`
- Random operators, those can be found under `numpy.random`

Totally, there are 497 operators generated.

| MXNet | ChainerX | JAX   |
| ----- | -------- | ----- |
| 17.8% | 23.8%    | 42.1% |

## Install

For users:

```
pip install git+https://github.com/hgt312/NumpyXBench
```

For developer (necessary for report generation):

```
git clone https://github.com/hgt312/NumpyXBench.git
cd NumpyXBench/
pip install -e .
```

## Report generation

### Install backends

1. Install MXNet from source:

   http://mxnet.incubator.apache.org/versions/master/install/ubuntu_setup.html

   In cmake flags, add `-DCMAKE_BUILD_TYPE=RELEASE`.

   With TVM support, add `-DUSE_TVM_OP=ON`.

2. Install Jax

https://github.com/google/jax#pip-installation

3. Install ChainerX

https://docs.chainer.org/en/stable/chainerx/install/index.html

### Install necessary dependencies

```
cd doc
pip install -r requirements.txt
```

### Build website

#### CPU only

```
CUDA_VISIBLE_DEVICES=-1 python -m NumpyXBench.tools --warmup 10 --runs 25 --device cpu --info "MacBook Pro, CPU"
sphinx-build -b html . _build/cpu -A current_device=CPU
```

#### With GPU enabled

```
CUDA_VISIBLE_DEVICES=-1 python -m NumpyXBench.tools --warmup 10 --runs 25 --device cpu --info "[Machine infomation]"
sphinx-build -b html . _build/cpu -A current_device=CPU
CUDA_VISIBLE_DEVICES=0 python -m NumpyXBench.tools --warmup 10 --runs 25 --device gpu --info "[Machine infomation]"
sphinx-build -b html . _build/gpu -A current_device=GPU
```

## Simple usage

Except generate reports in the website, this python package can be used to run some random benchmarks in CMD/jupyter notebook.

Note that you need to determine if gpu is visiable by set environment `CUDA_VISIBLE_DEVICES`. Then, before starting benchmark, run helper function to set default device.

```python
from NumpyXBench.utils import global_set_cpu, global_set_gpu

global_set_gpu()  # global_set_cpu()
```

1. Obtain an op from a toolkit which contains its default config

```python
# random config
from NumpyXBench.toolkits import add_toolkit

toolkit = add_toolkit
op = toolkit.get_operator_cls()('np')
config = toolkit.get_random_config_func('RealTypes')()
res = toolkit.get_benchmark_func()(op, config, 'forward')
```

```python
# determined config
from NumpyXBench.toolkits import broadcast_divide_toolkit

toolkit = broadcast_divide_toolkit
op = toolkit.get_operator_cls()('mx')
configs = toolkit.get_determined_config_func(['float32'])()
for c in configs:
    res = toolkit.get_benchmark_func()(op, c, 'backward', warmup=1, runs=10)
    print(res)
```

2. Another more flexible way.

```python
from NumpyXBench.operators import Add
from NumpyXBench.configs import get_random_size_config
from NumpyXBench.utils import run_binary_op_benchmark

op = Add(backend='numpy')
config = get_random_size_config(['float32'])
res = run_binary_op_benchmark(op, config, 'forward')
```

3. On multiple frameworks.

```python
from NumpyXBench.toolkits import add_toolkit
from NumpyXBench.utils import run_op_frameworks_benchmark

res = run_op_frameworks_benchmark(*add_toolkit.get_tools('AllTypes'), ['mx', 'np', 'chx', 'jax'], 'forward')
```

4. Test registered toolkits and brief visualization. The data generated from function `run_op_frameworks_benchmark` can be fed to `draw_one_plot`.

```python
from NumpyXBench.tools import test_all_operators, draw_one_plot, test_operators
from NumpyXBench import toolkits

res = test_operators([toolkits.mod_toolkit, toolkits.multiply_toolkit], is_random=False, dtypes=['float32'], times=6, warmup=3, runs=5)
# res = test_all_operators(is_random=False, dtypes=['float32'], times=6, warmup=1, runs=2)
draw_one_plot('mod', res['mod'], mode='note', info='mbp, cpu')  # use notebook to see the plot
```

5. Test coverage (only for frameworks that has same API with NumPy).

```python
from NumpyXBench.tools import test_numpy_coverage

res = test_numpy_coverage('jax')  # res = {'passed': [...], 'failed': [...]}
print(len(res['passed']) / (len(res['passed']) + len(res['failed'])))
```

## How to contribute

Refer to [Development Doc](doc.html).