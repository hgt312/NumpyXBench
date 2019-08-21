import math
import random

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
from numpy import random as nd

from .config_spaces import *

__all__ = ['get_random_shape_config', 'get_random_size_config', 'get_range_creation_config',
           'get_random_withaxis_config', 'get_size_configs']


def _gen_random_shape(ndim):
    high = math.ceil(math.pow(10**8, 1/ndim))
    shape = nd.randint(low=1, high=high, size=ndim)
    return tuple(shape)


def get_random_shape_config(dtypes):
    config_space = random_ndim_cs
    config = config_space.sample_configuration()
    shape = _gen_random_shape(config.get('ndim'))
    # random dtype
    config_space = cs.ConfigurationSpace()
    config_space.add_hyperparameter(csh.CategoricalHyperparameter('dtype', choices=dtypes))
    config = config_space.sample_configuration()
    dtype = config.get('dtype')
    return {'shape': shape, 'dtype': dtype}


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


def get_size_configs(dtypes):
    dtype = dtypes[0]
    result = [{'shape': (50,), 'dtype': dtype},
              {'shape': (5000,), 'dtype': dtype},
              {'shape': (50000,), 'dtype': dtype},
              {'shape': (100000,), 'dtype': dtype},
              {'shape': (1000000,), 'dtype': dtype},
              {'shape': (3000000,), 'dtype': dtype}]
    return result


def get_range_creation_config(op_name, dtypes):
    config_space = random_range_cs
    config = config_space.sample_configuration()
    config_dict = config.get_dictionary()
    config_dict.update({'stop': config_dict['start'] + config_dict.pop('interval')})
    # random dtype
    config_space = cs.ConfigurationSpace()
    config_space.add_hyperparameter(csh.CategoricalHyperparameter('dtype', choices=dtypes))
    config = config_space.sample_configuration()
    dtype = config.get('dtype')
    config_dict.update({'dtype': dtype})
    if op_name == 'linspace':
        config_space = random_num_cs
        config = config_space.sample_configuration()
        config_dict.update(config.get_dictionary())
    return config_dict


def get_random_withaxis_config(dtypes):
    config_space = random_ndim_cs
    config = config_space.sample_configuration()
    ndim = config.get('ndim')
    shape = _gen_random_shape(ndim)
    # random dtype
    config_space = cs.ConfigurationSpace()
    config_space.add_hyperparameter(csh.CategoricalHyperparameter('dtype', choices=dtypes))
    config = config_space.sample_configuration()
    dtype = config.get('dtype')
    # random axis
    axis = random.randint(0, ndim)
    axis = None if axis == ndim else axis
    return {'shape': shape, 'dtype': dtype, 'axis': axis}
