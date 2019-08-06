import math

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from numpy import random as nd

# __all__ = []


def _gen_random_shape(ndim):
    high = math.ceil(math.pow(10**9, 1/ndim))
    shape = nd.randint(low=1, high=high, size=ndim)
    return tuple(shape)


def get_binary_op_config():
    cs = CS.ConfigurationSpace()
    ndim = CSH.UniformIntegerHyperparameter('ndim', lower=2, upper=32, log=False)
    dtype = CSH.CategoricalHyperparameter('dtype', choices=['float32', 'float64'])
    cs.add_hyperparameters([ndim, dtype])
    config = cs.sample_configuration()
    shape = _gen_random_shape(config.get('ndim'))
    return {'shape': shape, 'dtype': config.get('dtype')}
