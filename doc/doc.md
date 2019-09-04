# Development Document

### Ops

We write configs for different categories of numpy operators, this can be obtain from [deepnumpy-doc](https://github.com/mli/deepnumpy-doc):

- Unary ops
- Binary ops
- Array creations
- …...

### Frameworks

We should support these frameworks, some has same API as NumPy, can be supported easily, while others may cost a lot of time to deal with:

- [NumPy](https://docs.scipy.org/doc/numpy/index.html)
- [MXNet](https://mxnet.apache.org/)
- [Chainerx & Cupy](https://docs.chainer.org/en/stable/chainerx/)
- [JAX](https://github.com/google/jax)
- torch (TBD)
- tensorflow 1&2 (TBD)

### Design

**Ops package:** in directory `operators`, ops with different numpy modules in different files, an Op object only need one argument, `backend`. All the ops under numpy should in `common_ops.py`, and linear algebra ops (under numpy.linalg) should be written in `la_ops.py` …...

In this part, most of ops can be generate by template. If the path to get the op function in each framework has the same/similar pattern (means that its user interface is same/similar to NumPy), it can be generated automatically.

If you want to add your custom op class, sample code is shown below, just return the function pointer of each backend, actually different operator can be returned:

```python
__all__ = ['MyRandomNormal']  # add op to here

class MyRandomNormal(CommonOp):
    def __init__(self, backend):
        super(MyRandomNormal, self).__init__(backend=backend)

    def get_forward_func(self, *args, **kwargs):
        if backend_switcher[self._backend] == "numpy":
            return numpy.random.normal
        elif backend_switcher[self._backend] == "jax.numpy":
            return jax.random.normal
        else:
            raise NotImplementedError("Backend not supported now!")
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

**Toolkits package:** Toolkits package register an operator, its name, its config-generation function and benchmark function. The toolkits is divided by operator type. In addition, one toolkits record the information such as if the operator has backward and the types it supported.

A sample toolkit:

```python
sum_toolkit = Toolkit(has_backward=True, name='sum', operator_cls=ops.Sum,
                      random_config_func=get_random_withaxis_config,
                      benchmark_func=run_unary_op_benchmark)
```

**Others:** need test, doc-gen, logging.

## Contribute

For all the operator list can be obtained, the mainly work is to write config generation functions and corresponding benchmark functions for lots of types of operators.
