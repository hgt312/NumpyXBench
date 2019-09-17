from functools import partial
import math
import random

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
from numpy import random as nd

from .config_spaces import *

__all__ = ['get_random_shape_config', 'get_random_size_config', 'get_random_withaxis_config', 'get_broadcast_configs',
           'get_size_configs', 'get_random_arange_config', 'get_random_linspace_config', 'get_size_axis_configs']


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
    configs = [{'shape': (1, 28, 28), 'dtype': dtype},
               {'shape': (64, 28, 28), 'dtype': dtype},
               {'shape': (32, 3, 224, 224), 'dtype': dtype},
               {'shape': (32, 224, 224, 3), 'dtype': dtype},
               {'shape': (64, 3, 224, 224), 'dtype': dtype},
               {'shape': (100, 100, 100, 10), 'dtype': dtype}]
    return configs


def get_broadcast_configs(dtypes):
    dtype = dtypes[0]
    configs = [{'shape1': (1, 28, 28), 'shape2': (1, 28), 'dtype': dtype},
               {'shape1': (64, 28, 28), 'shape2': (1, 28), 'dtype': dtype},
               {'shape1': (32, 3, 224, 224), 'shape2': (1, 224, 224), 'dtype': dtype},
               {'shape1': (32, 224, 224, 3), 'shape2': (1, 224, 3), 'dtype': dtype},
               {'shape1': (64, 3, 224, 224), 'shape2': (64, 3, 1, 1), 'dtype': dtype},
               {'shape1': (100, 100, 100, 10), 'shape2': (10, ), 'dtype': dtype}]
    return configs


def get_size_axis_configs(dtypes):
    dtype = dtypes[0]
    configs = [{'shape': (10, 10, 10), 'dtype': dtype, 'axis': None},
               {'shape': (10, 10, 10), 'dtype': dtype, 'axis': 2},
               {'shape': (100, 100, 10), 'dtype': dtype, 'axis': None},
               {'shape': (100, 100, 10), 'dtype': dtype, 'axis': 1},
               {'shape': (100, 100, 10), 'dtype': dtype, 'axis': 2},
               {'shape': (100, 100, 100), 'dtype': dtype, 'axis': 0}]
    return configs


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


def get_random_arange_config(dtypes):
    return get_range_creation_config('arange', dtypes)


def get_random_linspace_config(dtypes):
    return get_range_creation_config('linspace', dtypes)


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
