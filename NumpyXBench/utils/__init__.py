import chainerx
import mxnet

from .benchmarks import *


def global_set_gpu():
    mxnet.test_utils.set_default_context(mxnet.gpu(0))
    chainerx.set_default_device('cuda:0')


def global_set_cpu():
    mxnet.test_utils.set_default_context(mxnet.cpu())
    chainerx.set_default_device('native')
