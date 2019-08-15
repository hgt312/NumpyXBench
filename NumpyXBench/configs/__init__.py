import math

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
from numpy import random as nd

__all__ = ['get_random_shape_config', 'get_random_size_config', 'get_range_creation_config']


def _gen_random_shape(ndim):
    high = math.ceil(math.pow(10**9, 1/ndim))
    shape = nd.randint(low=1, high=high, size=ndim)
    return tuple(shape)


def get_random_shape_config(dtypes):
    config_space = cs.ConfigurationSpace()
    ndim = csh.UniformIntegerHyperparameter('ndim', lower=2, upper=32, log=False)
    dtype = csh.CategoricalHyperparameter('dtype', choices=dtypes)
    config_space.add_hyperparameters([ndim, dtype])
    config = config_space.sample_configuration()
    shape = _gen_random_shape(config.get('ndim'))
    return {'shape': shape, 'dtype': config.get('dtype')}


def get_random_size_config(dtypes):
    config_space = cs.ConfigurationSpace()
    size = csh.UniformIntegerHyperparameter('size', lower=1, upper=4096**2, log=True)
    dtype = csh.CategoricalHyperparameter('dtype', choices=dtypes)
    config_space.add_hyperparameters([size, dtype])
    config = config_space.sample_configuration()
    shape = (config.get('size'),)
    return {'shape': shape, 'dtype': config.get('dtype')}


def get_range_creation_config(op_name, dtypes):
    config_space = cs.ConfigurationSpace()
    start = csh.UniformIntegerHyperparameter('start', lower=0, upper=int(1e4))
    interval = csh.UniformIntegerHyperparameter('interval', lower=5, upper=int(1e5), log=True)
    dtype = csh.CategoricalHyperparameter('dtype', choices=dtypes)
    if op_name == 'linspace':
        num = csh.UniformIntegerHyperparameter('num', lower=5, upper=int(1e5), log=True)
        config_space.add_hyperparameters([start, interval, num, dtype])
    else:
        config_space.add_hyperparameters([start, interval, dtype])
    config = config_space.sample_configuration()
    config_dict = config.get_dictionary()
    config_dict.update({'stop': config_dict['start'] + config_dict.pop('interval')})
    return config_dict
