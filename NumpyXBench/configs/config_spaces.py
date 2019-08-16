import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh

from ..utils.common import *

__all__ = ['random_ndim_cs', 'random_size_cs', 'random_range_cs', 'random_num_cs']

ndim = csh.UniformIntegerHyperparameter('ndim', lower=2, upper=6, log=False)
size = csh.UniformIntegerHyperparameter('size', lower=1, upper=5000**2, log=True)
num = csh.UniformIntegerHyperparameter('num', lower=5, upper=int(1e5), log=True)
start = csh.UniformIntegerHyperparameter('start', lower=0, upper=int(1e4))
interval = csh.UniformIntegerHyperparameter('interval', lower=5, upper=int(1e5), log=True)

random_ndim_cs = cs.ConfigurationSpace()
random_ndim_cs.add_hyperparameters([ndim])

random_size_cs = cs.ConfigurationSpace()
random_size_cs.add_hyperparameters([size])

random_range_cs = cs.ConfigurationSpace()
random_range_cs.add_hyperparameters([start, interval])

random_num_cs = cs.ConfigurationSpace()
random_num_cs.add_hyperparameters([num])
