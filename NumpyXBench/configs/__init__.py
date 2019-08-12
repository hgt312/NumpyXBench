import math

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
from numpy import random as nd

__all__ = ['get_random_shape_config', 'get_random_size_config']

float_dtypes = csh.CategoricalHyperparameter('dtype', choices=['float32', 'float64'])


def _gen_random_shape(ndim):
    high = math.ceil(math.pow(10**9, 1/ndim))
    shape = nd.randint(low=1, high=high, size=ndim)
    return tuple(shape)


def get_random_shape_config():
    config_space = cs.ConfigurationSpace()
    ndim = csh.UniformIntegerHyperparameter('ndim', lower=2, upper=32, log=False)
    config_space.add_hyperparameters([ndim, float_dtypes])
    config = config_space.sample_configuration()
    shape = _gen_random_shape(config.get('ndim'))
    return {'shape': shape, 'dtype': config.get('dtype')}


def get_random_size_config():
    config_space = cs.ConfigurationSpace()
    size = csh.UniformIntegerHyperparameter('size', lower=1, upper=4096**2, log=True)
    config_space.add_hyperparameters([size, float_dtypes])
    config = config_space.sample_configuration()
    shape = (config.get('size'),)
    return {'shape': shape, 'dtype': config.get('dtype')}
