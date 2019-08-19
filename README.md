# Benchmark for NumPy Compatible Operators

[doc](./doc/doc.md)

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

`pip install git+https://github.com/hgt312/NumpyXBench`

For developer:

```
git clone https://github.com/hgt312/NumpyXBench.git
cd NumpyXBench/
pip install -e .
```

## Simple usage

1. Obtain an op from a blob which contains its default config

```python
from NumpyXBench.blobs import get_ones_blob

blob, _ = get_ones_blob('RealTypes')
op = blob[0](backend='np')
config = blob[1]()
res = blob[2](op, config, 'both')
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
from NumpyXBench.blobs import get_add_blob
from NumpyXBench.utils import run_op_frameworks_benchmark

res = run_op_frameworks_benchmark(*get_add_blob()[0], ['mx', 'np', 'chx', 'jax'], 'forward')
```

4. Test all registered blobs.

```python
from NumpyXBench.tools import test_all_blobs

res = test_all_blobs()
```

5. Test coverage (only for frameworks that has same API with NumPy).

```python
from NumpyXBench.tools import test_numpy_coverage

res = test_numpy_coverage('jax')  # res = {'passed': [...], 'failed': [...]}
print(len(res['passed']) / (len(res['passed']) + len(res['failed'])))
```

## How to contribute

TODO