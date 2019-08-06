### Ops

By numpy categories of operators, this can be obtain from [deepnumpy-doc](https://github.com/mli/deepnumpy-doc):

- Unary ops
- Binary ops
- Array creations
- …...

### Frameworks

- numpy
- mxnet
- chainerx & cupy
- jax
- torch

### API & Design

**Sample code:**

```python
from NumpyXBench.operators import Add
from NumpyXBench.configs import get_binary_op_config
from NumpyXBench.utils import run_binary_op_benchmark

op = Add(backend='numpy')
config = get_binary_op_config()
res = run_binary_op_benchmark(op, config, 'forward')
```

**Sample directory tree:** 

```
.
├── NumpyXBench
│   ├── __init__.py
│   ├── configs
│   │   ├── __init__.py
│   ├── operators
│   │   ├── __init__.py
│   │   └── binary_ops.py
│   └── utils
│       ├── __init__.py
├── lincense
├── README.md
├── samples
└── tests
```

**Op package:** in directory `operators`, ops with different types in different files, an Op object only need one argument, `backend`.

**Config package:** get information of input arguments: **input shape, dtype …...**

**Utils:** functions to do benchmarks (**single op, op cross frameworks, ops on single framework, and ops on frameworks**).

**Metrics:** TBD

**Others:** need things to collect necessary information for each operator, register in op class/store in file or database
