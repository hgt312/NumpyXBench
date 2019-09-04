## What does this project do?

This is a project used to benchmark the operators (functions) of the python librarieswhich have compatible API with [Numpy](https://docs.scipy.org/doc/numpy/index.html), now it can generate some reports for operators in [MXNet](https://mxnet.apache.org/) (new numpy programming style), [ChainerX](https://docs.chainer.org/en/stable/chainerx/) and [JAX](https://github.com/google/jax).

## Operator coverage

I divide opertors into several categories: 

- Common operators, those can be found under `numpy`
- FFT operators, those can be found under `numpy.fft`
- Linear algebra operators, those can be found under `numpy.linalg`
- Random operators, those can be found under `numpy.random`

| MXNet (numpy branch) | ChainerX | JAX   |
| -------------------- | -------- | ----- |
| 17.0%                | 23.8%    | 91.2% |

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

1. Install MXNet from source (switch to numpy branch):

   http://mxnet.incubator.apache.org/versions/master/install/ubuntu_setup.html

2. Install Jax

   ```
   # CPU-only version
   pip install --upgrade jax jaxlib
   
   # with GPU supported
   PYTHON_VERSION=cp37  # alternatives: cp27, cp35, cp36, cp37
   CUDA_VERSION=cuda92  # alternatives: cuda90, cuda92, cuda100
   PLATFORM=linux_x86_64  # alternatives: linux_x86_64
   BASE_URL='https://storage.googleapis.com/jax-releases'
   pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.23-$PYTHON_VERSION-none-$PLATFORM.whl
   
   pip install --upgrade jax  # install jax
   ```

3. Install ChainerX

   ```
   # CPU-only version
   export CHAINER_BUILD_CHAINERX=1
   export MAKEFLAGS=-j8  # Using 8 parallel jobs.
   pip install --pre chainer
   
   # with GPU supported
   export CHAINER_BUILD_CHAINERX=1
   export CHAINERX_BUILD_CUDA=1
   export CUDNN_ROOT_DIR=path/to/cudnn
   export MAKEFLAGS=-j8  # Using 8 parallel jobs.
   pip install --pre cupy
   pip install --pre chainer
   ```

### Install necessary dependencies

```
cd doc
pip install -r requirements.txt
```

### Build website

#### CPU only

```
python -m NumpyXBench.tools  # -h for help
sphinx-build -b html . _build -A current_device=CPU
```

#### With GPU enabled

```
python -m NumpyXBench.tools  # -h for help
sphinx-build -b html . _build -A current_device=CPU
sphinx-build -b html . _build/gpu -A current_device=GPU
```

## Simple usage

1. Obtain an op from a toolkit which contains its default config

```python
from NumpyXBench.toolkits import add_toolkit

toolkit = add_toolkit
op = toolkit.get_operator_cls()('np')
config = toolkit.get_random_config_func('RealTypes')()
res = toolkit.get_benchmark_func()(op, config, 'forward')
```

2. Another more flexible way.

```python
from NumpyXBench.operators import Add
from NumpyXBench.configs import get_random_size_config
from NumpyXBench.utils import run_binary_op_benchmark

op = Add(backend='numpy')
config = get_random_size_config()
res = run_binary_op_benchmark(op, config, 'forward')
```

3. On multiple frameworks.

```python
from NumpyXBench.toolkits import add_toolkit
from NumpyXBench.utils import run_op_frameworks_benchmark

res = run_op_frameworks_benchmark(*add_toolkit.get_tools('AllTypes'), ['mx', 'np', 'chx', 'jax'], 'forward')
```

4. Test all registered toolkits and brief visualization.

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