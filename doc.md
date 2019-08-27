# Document

### Ops

We write configs for different categories of numpy operators, this can be obtain from [deepnumpy-doc](https://github.com/mli/deepnumpy-doc):

- Unary ops
- Binary ops
- Array creations
- …...

### Frameworks

We should support these frameworks:

- numpy
- mxnet
- chainerx & cupy
- jax
- torch
- tensorflow 1&2 (TBD)

### Design

**Sample directory tree:** 

```
.
├── NumpyXBench
│   ├── __init__.py
│   ├── blobs
│   │   └── __init__.py
│   ├── configs
│   │   └── __init__.py
│   ├── operators
│   │   ├── __init__.py
│   │   ├── common_ops.py
│   │   └── la_ops.py
│   └── utils
│       └── __init__.py
├── doc.md
└── samples
```

**Blob package:** Blob package simply store an op and its default config-gen function and default benchmark function in a python tuple. 

For example, `add_blob = (ops.Add, get_random_size_config, run_binary_op_benchmark)`.

**Op package:** in directory `operators`, ops with different numpy modules in different files, an Op object only need one argument, `backend`. All the ops under numpy should in `common_ops.py`, and linear algebra ops (under numpy.linalg) should be written in `la_ops.py` …...

In this part, most of ops can be generate by template. If the path to get the op function in each framework has the same pattern, we can add it by simply append its name in a list. Sample code is shown below, look [code](NumpyXBench/operators/common_ops.py) to get details.

```python
common_op_list = ['add', 'subtract', 'multiply', 'divide'] # add op to here

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
```

**Config package:** get information of input arguments: **input shape, dtype** …… It depends on the type of op. Need to determine benchmark ways (random size/several determined size).

**Utils:** functions to do benchmarks (**single op, op cross frameworks, ops on single framework, and ops on frameworks**).

**Metrics:** speed, coverage (necessary), others TBD

**Others:** need test, doc-gen, logging, report-gen, and visualization.