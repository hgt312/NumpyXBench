# Development Document

## Ops

We write configs for different categories of numpy operators, this can be obtain from [deepnumpy-doc](https://github.com/mli/deepnumpy-doc):

- Unary ops
- Binary ops
- Array creations
- …...

## Frameworks

We should support these frameworks, some has same API as NumPy, can be supported easily, while others may cost a lot of time to deal with:

- [NumPy](https://docs.scipy.org/doc/numpy/index.html)
- [MXNet](https://mxnet.apache.org/)
- [Chainerx & Cupy](https://docs.chainer.org/en/stable/chainerx/)
- [JAX](https://github.com/google/jax)
- torch (TBD)
- tensorflow 1&2 (TBD)

## Design

**Ops package:** in directory `operators`, ops with different numpy modules in different files, an Op object only need one argument, `backend`. All the ops under numpy should in `common_ops.py`, and linear algebra ops (under numpy.linalg) should be written in `la_ops.py` …...

In this part, most of ops can be generate by template. If the path to get the op function in each framework has the same/similar pattern (means that its user interface is same/similar to NumPy), it can be generated automatically.

If you want to add your custom op class, sample code is shown below, just return the function pointer of each backend, actually different operator can be returned:

```python
__all__ = ['TVMAdd']  # add op to here

class TVMAdd(CommonOp):
    _name = "tvm_add"
    def __init__(self, backend):
        super(TVMAdd, self).__init__(backend=backend)

    def get_forward_func(self, *args, **kwargs):
        if backend_switcher[self._backend] == "numpy":
            return numpy.add
        elif backend_switcher[self._backend] == "jax.numpy":
            return jax.add
        elif backend_switcher[self._backend] == "mxnet.numpy":
            return mxnet.numpy.add
        elif backend_switcher[self._backend] == "chainerx":
            return chainerx.add
```

**Configs package:** get information of input arguments: **input shape, dtype** …… It depends on the type of op. Need to determine benchmark ways (random size/several determined size). A big config function can be built by multiple small provided config spaces. Random config would be a python dict, while the determined would be a list.

**Utils:** 

- Functions to do benchmarks (**single op, op cross frameworks, ops on single framework, and ops on frameworks**).
- Helper functions for backends.
- Time metric.

**Tools:**

- Tools for global setting
- Tool for test coverage
- Tools for generate result plots and website generation

**Toolkits package:** Toolkits package register an operator, its config-generation function and benchmark function. The toolkits is divided by operator type. In addition, one toolkits record the information such as if the operator has backward and the types it supported.

A sample toolkit:

```python
sum_toolkit = Toolkit(has_backward=True, operator_cls=ops.Sum,
                      random_config_func=get_random_withaxis_config,
                      benchmark_func=run_unary_op_benchmark)
```

**Others:** need test, doc-gen, logging.

## Contribute

For all the operator list can be obtained, the mainly work is to write config generation functions and corresponding benchmark functions for lots of types of operators. After that, register the operator so that it can be seen in generated report.

### Step 1, write config generation functions:

A config is consist of parameter used to generate input tensor and other necessary arguments. A single config is a python dict, for example `{'shape': (10, 20), 'dtype': 'float32'}`.

The input of a config generation function is always `dtype`. To generate random configs, return a dict; to generate determined configs, return a list of dicts.

```python
# random
def get_random_size_config(dtypes):
    config_space = random_size_cs
    config = config_space.sample_configuration()
    shape = (config.get('size'),)
    # random dtype
    config_space = cs.ConfigurationSpace()
    config_space.add_hyperparameter(csh.CategoricalHyperparameter('dtype', choices=dtypes))
    config = config_space.sample_configuration()
    dtype = config.get('dtype')
    return {'shape': shape, 'dtype': dtype}


# determined
def get_size_configs(dtypes):
    dtype = dtypes[0]
    configs = [{'shape': (1, 28, 28), 'dtype': dtype},
               {'shape': (64, 28, 28), 'dtype': dtype},
               {'shape': (32, 3, 224, 224), 'dtype': dtype},
               {'shape': (32, 224, 224, 3), 'dtype': dtype},
               {'shape': (64, 3, 224, 224), 'dtype': dtype},
               {'shape': (100, 100, 100, 10), 'dtype': dtype}]
    return configs
```

*P.S.: There are some basic config spaces that can be useful for writing config functions.*

### Step 2, write benchmark functions:

A benchmark function defines how to test the performance of a specific operator, its implementation is related to the config corresponding to the operator.

Handle different part of config by different methods: handle tensors by a python list, then handle other arguments by a python dict. A general benchmark function which can handle all configs with tensors with same shape and dtype is completed, it could be a good reference.

### Step 3, register the operator:

A toolkit tells an operator's corresponding python class, if has backward, support types, config generate functions and benchmark function. A sample toolkit is shown below, whole the parameter list can be read in source code.

1. Write a toolkit, then add it to `__all__`
2. Add it to `doc/report.rst`

```python
multiply_toolkit = Toolkit(has_backward=True, operator_cls=ops.Multiply,
                           random_config_func=get_random_size_config,
                           determined_config_func=get_size_configs,
                           benchmark_func=run_binary_op_benchmark)
```

